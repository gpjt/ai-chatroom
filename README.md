# ai-chatroom

A Telegram bot where you can have multiple OpenAI- or Anthropic-compatible AIs chiming in.
Code originally written by Claude <https://claude.ai/> but since heavily
modified.

See [this blog post](https://www.gilesthomas.com/2024/12/ai-chatroom-1) for
the background.

Very much a WIP -- don't expect it to work well yet!

To use, copy the `creds.json.sample` file and fill in the credentials; each one
has a description of what to put into it.  You must provide at least one AI API
key in the `provider_api_keys` dict -- you can see the valid options in the file
providers.json.
