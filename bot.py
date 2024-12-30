import json
import logging
import random
from pathlib import Path
from typing import List

import aiohttp

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


BASE_DIR = Path(__file__).resolve().parent
CREDS_FILE = BASE_DIR / "creds.json"
PROVIDER_CONFIG_FILE = BASE_DIR / "providers.json"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_creds():
    with open(CREDS_FILE) as f:
        creds = json.load(f)
    required = ["telegram_bot_token", "bot_secret_key", "provider_api_keys"]
    missing = [var for var in required if var not in creds]
    if missing:
        raise EnvironmentError(f"Your creds file is missing the following: {', '.join(missing)}")
    if type(creds["provider_api_keys"]) != dict:
        raise EnvironmentError("Please provide a mapping from providers to API keys for `provider_api_keys`")
    if len(creds["provider_api_keys"]) == 0:
        raise EnvironmentError("Please provide at least one API key in `provider_api_keys`")
    return creds



def _create_system_prompt(ai_identifier):
    return f"""You are in a chat session with one or more humans, and potentially other AIs.
    Messages from humans are identified by 👤[Name], messages from AIs that are not you are identified by 🤖[Name],
    and your own messages are identified by {ai_identifier}.  This applies only to the context that you
    are sent, you MUST NOT prefix your own responses with {ai_identifier}.
    After each message from a human, all participating AIs will be offered a chance to respond in a random order.
    Once all have responded, they will be offered a chance to respond again so that they can answer any points
    raised by the other AIs. You can choose not to respond by saying just 'PASS'.
    You are welcome to address in your responses anything raised by either the humans or any other AIs.
    You should keep your response to less than 1024 tokens."""



class AIProvider:
    def __init__(self, name, api_key, base_url, model):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = _create_system_prompt(f"🤖[{name}]")


    async def make_request(self, messages):
        async with aiohttp.ClientSession() as session:
            headers = self.get_headers()
            payload = self.get_payload(messages)
            try:
                logging.info(
                    f"Making request to {self.base_url}\n"
                    f"   Headers: {headers}\n"
                    f"   JSON: {payload}\n"
                )
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        response = self.parse_response(data)
                        logging.info(f"Got response: {response}")
                        return response
                    else:
                        return f"Error: {await response.text()}"
            except Exception as e:
                return f"Error making request to {self.name}: {str(e)}"


class OpenAIProvider(AIProvider):
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }


    def get_payload(self, messages):
        return {
            "messages": [{"role": "system", "content": self.system_prompt}] + messages,
            "model": self.model,
            "temperature": 0.7,
        }

    def parse_response(self, data):
        return data['choices'][0]['message']['content']



class AnthropicProvider(AIProvider):
    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def get_payload(self, messages):
        return {
            "system": self.system_prompt,
            "messages": messages,
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 1024,
        }

    def parse_response(self, data):
        if len(data["content"]) == 0:
            return "PASS"
        return data["content"][0]["text"]


def build_providers(provider_api_keys):
    with open(PROVIDER_CONFIG_FILE) as f:
        ai_provider_configs = json.load(f)

    providers = {}
    for name, config in ai_provider_configs.items():
        api_key = provider_api_keys.get(name)
        if api_key:
            if config.get("api_type") == "openai":
                provider_class = OpenAIProvider
            elif config.get("api_type") == "anthropic":
                provider_class = AnthropicProvider
            else:
                raise Exception(f'Unknown api_type {config.get("api_type")!r} for {name}')

            providers[name] = provider_class(
                name,
                api_key,
                config["base_url"],
                config["model"],
            )
    if len(providers) == 0:
        raise EnvironmentError("No API keys for a valid provider found")
    return providers


class AIChat:
    def __init__(self, providers):
        self.providers = providers
        self.chat_history = []


    async def process_message(self, chat_id: int, user_name: str, message_text: str) -> List[str]:
        # Format the user message
        formatted_message = f"👤[{user_name}]: {message_text}"

        self.chat_history += [
            {"role": "user", "content": formatted_message}
        ]

        ai_order = list(self.providers.values())
        random.shuffle(ai_order)

        have_response = False
        for provider in ai_order:
            response = await provider.make_request(self.chat_history)
            if response.strip().upper() != "PASS":
                formatted_response = f"🤖[{provider.name}]: {response}"
                yield formatted_response
                self.chat_history.append({"role": "assistant", "content": formatted_response})
                have_response = True

        # Second round: Allow AIs to respond to each other, if at least one of them
        # replied
        if have_response:
            random.shuffle(ai_order)  # Randomize order again for second round
            for provider in ai_order:
                response = await provider.make_request(self.chat_history)
                if response.strip().upper() != "PASS":
                    formatted_response = f"🤖[{provider.name}]: {response}"
                    yield formatted_response
                    self.chat_history.append({"role": "assistant", "content": formatted_response})


class TelegramBot:
    def __init__(self, token, secret_key, providers):
        self.application = Application.builder().token(token).build()
        self.providers = providers
        self.authorized_chats = {}
        self.secret_key = secret_key

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        # Check if the chat is already authorized
        if chat_id in self.authorized_chats:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat is already authorized and initialized."
            )
            return

        # Check for secret key
        if not context.args or len(context.args) != 1 or context.args[0] != self.secret_key:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Unauthorized. Please provide the correct secret key."
            )
            return

        # Authorize and initialize the chat
        ai_chat = AIChat(providers=self.providers)
        self.authorized_chats[chat_id] = ai_chat
        await context.bot.send_message(
            chat_id=chat_id,
            text="Chat authorized and initialized."
        )


    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        chat_id = update.effective_chat.id

        # Check if the chat is authorized
        if chat_id not in self.authorized_chats:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat is not authorized. Use /start with the secret key to begin."
            )
            return

        user_name = update.effective_user.first_name
        message_text = update.message.text

        chat = self.authorized_chats[chat_id]
        async for response in chat.process_message(chat_id, user_name, message_text):
            await context.bot.send_message(
                chat_id=chat_id,
                text=response
            )


def main():
    creds = load_creds()

    providers = build_providers(creds["provider_api_keys"])

    bot = TelegramBot(token=creds["telegram_bot_token"], secret_key=creds["bot_secret_key"], providers=providers)
    bot.application.run_polling()

if __name__ == "__main__":
    main()