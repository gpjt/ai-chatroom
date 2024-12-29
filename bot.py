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
        "base_url": "https://api.anthropic.com/v1/messages"
    },
    "GPT": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions"
    },
    "Grok": {
        "env_key": "GROK_API_KEY",
        "base_url": "https://api.x.ai/v1/chat/completions"
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions"
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
    def __init__(self, name: str, api_key: str, base_url: str, system_prompt: str):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt

class AIChat:
    def __init__(self):
        # Initialize providers based on available API keys
        self.providers: Dict[str, AIProvider] = {}

        for name, config in API_CONFIGS.items():
            api_key = os.getenv(config["env_key"])
            if api_key:
                self.providers[name] = AIProvider(
                    name,
                    api_key,
                    config["base_url"],
                    self._create_system_prompt(f"🤖[{name}]")
                )

        # Store active chats and their configurations
        self.active_chats: Dict[int, List[str]] = {}  # chat_id -> list of active AI names

    def _create_system_prompt(self, ai_identifier: str) -> str:
        return f"""You are in a chat session with one or more humans, and potentially other AIs.
        Messages from humans are identified by 👤[Name], messages from AIs that are not you are identified by 🤖[Name],
        and your own messages are identified by {ai_identifier}.
        After each message from a human, all participating AIs will be offered a chance to respond in a random order.
        Once all have responded, they will be offered a chance to respond again so that they can answer any points
        raised by the other AIs. You can choose not to respond by saying just 'PASS'.
        You are welcome to address in your responses anything raised by either the humans or any other AIs."""

    async def _make_ai_request(self, provider: AIProvider, messages: List[Dict[str, str]]) -> str:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "messages": messages,
                "model": "gpt-4",  # Adjust model name as needed for each provider
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
        if chat_id not in self.active_chats:
            return ["This chat has no active AI participants. Use /start to begin."]

        # Format the user message
        formatted_message = f"👤[{user_name}]: {message_text}"

        # Prepare conversation history (you'll need to implement history tracking)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_message}
        ]

        # First round: Get initial responses from all AIs in random order
        responses = []
        ai_order = list(self.active_chats[chat_id])
        random.shuffle(ai_order)

        for ai_name in ai_order:
            provider = self.providers[ai_name]
            response = await self._make_ai_request(provider, messages)
            if response.strip().upper() != "PASS":
                responses.append(f"🤖[{ai_name}]: {response}")

        # Second round: Allow AIs to respond to each other
        if len(responses) > 1:  # Only do second round if there were multiple responses
            second_round_messages = messages + [{"role": "assistant", "content": r} for r in responses]

            random.shuffle(ai_order)  # Randomize order again for second round
            for ai_name in ai_order:
                provider = self.providers[ai_name]
                response = await self._make_ai_request(provider, second_round_messages)
                if response.strip().upper() != "PASS":
                    responses.append(f"🤖[{ai_name}] (follow-up): {response}")

        return responses

class TelegramBot:
    def __init__(self, token: str):
        self.application = Application.builder().token(token).build()
        self.ai_chat = AIChat()
        self.authorized_chats = set()  # Store authorized chat IDs
        self.secret_key = os.getenv('BOT_SECRET_KEY')

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("add_ai", self.add_ai_command))
        self.application.add_handler(CommandHandler("remove_ai", self.remove_ai_command))
        self.application.add_handler(CommandHandler("list_ai", self.list_ai_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Initialize a new chat session with authentication"""
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
        self.authorized_chats.add(chat_id)
        self.ai_chat.active_chats[chat_id] = []
        self.ai_chat.chat_history[chat_id] = []  # Initialize empty history for new chat
        await context.bot.send_message(
            chat_id=chat_id,
            text="Chat authorized and initialized. Use /add_ai to add AI participants."
        )

    async def add_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add an AI to the chat"""
        chat_id = update.effective_chat.id

        # Check if the chat is authorized
        if chat_id not in self.authorized_chats:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat is not authorized. Use /start with the secret key to begin."
            )
            return
        if not context.args:
            available_ais = ", ".join(self.ai_chat.providers.keys())
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Please specify an AI to add. Available AIs: {available_ais}"
            )
            return

        ai_name = context.args[0]
        if ai_name not in self.ai_chat.providers:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Unknown AI: {ai_name}"
            )
            return

        if chat_id not in self.ai_chat.active_chats:
            self.ai_chat.active_chats[chat_id] = []

        if ai_name not in self.ai_chat.active_chats[chat_id]:
            self.ai_chat.active_chats[chat_id].append(ai_name)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Added {ai_name} to the chat"
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{ai_name} is already in the chat"
            )

    async def remove_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Remove an AI from the chat"""
        chat_id = update.effective_chat.id

        # Check if the chat is authorized
        if chat_id not in self.authorized_chats:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat is not authorized. Use /start with the secret key to begin."
            )
            return
        if not context.args:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Please specify an AI to remove"
            )
            return

        ai_name = context.args[0]
        if chat_id in self.ai_chat.active_chats and ai_name in self.ai_chat.active_chats[chat_id]:
            self.ai_chat.active_chats[chat_id].remove(ai_name)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Removed {ai_name} from the chat"
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"{ai_name} is not in the chat"
            )

    async def list_ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all AIs in the chat"""
        chat_id = update.effective_chat.id

        # Check if the chat is authorized
        if chat_id not in self.authorized_chats:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat is not authorized. Use /start with the secret key to begin."
            )
            return
        if chat_id in self.ai_chat.active_chats:
            ais = ", ".join(self.ai_chat.active_chats[chat_id]) if self.ai_chat.active_chats[chat_id] else "No AIs"
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"AIs in this chat: {ais}"
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="This chat has not been initialized. Use /start first."
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

        responses = await self.ai_chat.process_message(chat_id, user_name, message_text)

        for response in responses:
            await context.bot.send_message(
                chat_id=chat_id,
                text=response
            )

def main():
    # Validate environment variables before starting
    validate_env_vars()

    # Initialize and run the bot
    bot = TelegramBot(os.getenv('TELEGRAM_BOT_TOKEN'))
    bot.application.run_polling()

if __name__ == "__main__":
    main()