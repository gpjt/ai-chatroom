# ai-chatroom

A Telegram bot where you can have multiple OpenAI- or Anthropic-compatible AIs chiming in.
Code originally written by Claude <https://claude.ai/> but since heavily
modified.

See [this blog post](https://www.gilesthomas.com/2024/12/ai-chatroom-1) for
the background.

Very much a WIP -- don't expect it to work well yet!

To use, you *must* set the following environment variables:

```bash
export TELEGRAM_BOT_TOKEN=...
export BOT_SECRET_KEY=...
```

...and then one or more of the following:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GROK_API_KEY=...
export DEEPSEEK_API_KEY=...
```

The `TELEGRAM_BOT_TOKEN` is the one you get from the BotFather on Telegram.  The
`BOT_SECRET_KEY` is something you should make up yourself -- it cannot contain
spaces, and is something you provide to the `/start` command when you connect
a chat to the bot so that other people can't use your instance (and thus drain
your API credits).  The `_API_KEY` variables are just the ones you get from the
AI API providers.
