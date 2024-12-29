import logging
import os
import random
from typing import Dict, List

import aiohttp

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Required environment variables
REQUIRED_ENV_VARS = [
    'TELEGRAM_BOT_TOKEN',
    'BOT_SECRET_KEY'
]

# Optional API configurations
API_CONFIGS = {
    "Claude": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
        "model": "???",
    },
    "GPT": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
    },
    "Grok": {
        "env_key": "GROK_API_KEY",
        "base_url": "https://api.x.ai/v1/chat/completions",
        "model": "???",
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "???",
    }
}

def validate_env_vars():
    """Validate required environment variables and ensure at least one AI provider is available"""
    # Check required vars
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please ensure all required variables are set in your environment."
        )

    # Check if at least one AI provider is configured
    available_providers = [
        name for name, config in API_CONFIGS.items()
        if os.getenv(config["env_key"])
    ]

    if not available_providers:
        raise EnvironmentError(
            "No AI providers configured. Please set at least one of these environment variables:\n"
            + "\n".join([f"- {config['env_key']}" for config in API_CONFIGS.values()])
        )

class AIProvider:
    def __init__(self, name: str, api_key: str, base_url: str, system_prompt: str, model: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.model = model


def _create_system_prompt(ai_identifier):
    return f"""You are in a chat session with one or more humans, and potentially other AIs.
    Messages from humans are identified by 👤[Name], messages from AIs that are not you are identified by 🤖[Name],
    and your own messages are identified by {ai_identifier}.  This applies only to the context that you
    are sent, you MUST NOT prefix your own responses with {ai_identifier}.
    After each message from a human, all participating AIs will be offered a chance to respond in a random order.
    Once all have responded, they will be offered a chance to respond again so that they can answer any points
    raised by the other AIs. You can choose not to respond by saying just 'PASS'.
    You are welcome to address in your responses anything raised by either the humans or any other AIs."""


def build_providers():
    providers = {}
    for name, config in API_CONFIGS.items():
        api_key = os.getenv(config["env_key"])
        if api_key:
            providers[name] = AIProvider(
                name,
                api_key,
                config["base_url"],
                _create_system_prompt(f"🤖[{name}]"),
                config["model"],
            )
    return providers


class AIChat:
    def __init__(self, providers):
        self.providers = providers
        self.chat_history: Dict[int, List[str]] = {}  # chat_id -> history


    async def _make_ai_request(self, provider: AIProvider, messages: List[Dict[str, str]]) -> str:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "messages": messages,
                "model": provider.model,
                "temperature": 0.7
            }

            try:
                logging.info(
                    f"Making request to {provider.base_url}\n"
                    f"   Headers: {headers}\n"
                    f"   JSON: {payload}\n"
                )
                async with session.post(provider.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        response = data['choices'][0]['message']['content']
                        logging.info(f"Got response: {response}")
                        return response
                    else:
                        return f"Error: {await response.text()}"
            except Exception as e:
                return f"Error making request to {provider.name}: {str(e)}"

    async def process_message(self, chat_id: int, user_name: str, message_text: str) -> List[str]:
        # Format the user message
        formatted_message = f"👤[{user_name}]: {message_text}"

        # Prepare conversation history (you'll need to implement history tracking)
        messages = [
            {"role": "user", "content": formatted_message}
        ]

        # First round: Get initial responses from all AIs in random order
        responses = []
        ai_order = list(self.providers.values())
        random.shuffle(ai_order)

        for provider in ai_order:
            response = await self._make_ai_request(provider, [{"role": "system", "content": provider.system_prompt}] + messages)
            if response.strip().upper() != "PASS":
                responses.append(f"🤖[{provider.name}]: {response}")

        # Second round: Allow AIs to respond to each other
        if len(responses) > 1:  # Only do second round if there were multiple responses
            second_round_messages = messages + [{"role": "assistant", "content": r} for r in responses]

            random.shuffle(ai_order)  # Randomize order again for second round
            for provider in ai_order:
                response = await self._make_ai_request(provider, second_round_messages)
                if response.strip().upper() != "PASS":
                    responses.append(f"🤖[{provider.name}] (follow-up): {response}")

        return responses

class TelegramBot:
    def __init__(self, token, providers):
        self.application = Application.builder().token(token).build()
        self.providers = providers
        self.authorized_chats = {}
        self.secret_key = os.getenv('BOT_SECRET_KEY')

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
        ai_chat.chat_history[chat_id] = []
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

        responses = await self.authorized_chats[chat_id].process_message(chat_id, user_name, message_text)

        for response in responses:
            await context.bot.send_message(
                chat_id=chat_id,
                text=response
            )


def main():
    validate_env_vars()

    providers = build_providers()

    bot = TelegramBot(token=os.getenv('TELEGRAM_BOT_TOKEN'), providers=providers)
    bot.application.run_polling()

if __name__ == "__main__":
    main()