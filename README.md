# ai-chatroom

A Telegram bot where you can have multiple OpenAI- or Anthropic-compatible AIs chiming in.
Code originally written by Claude <https://claude.ai/> but since heavily
modified.

See [this blog post](https://www.gilesthomas.com/2024/12/ai-chatroom-1) for
the background.

This is not production-grade code! It works for me and I use it, but might not work for you.  Suggestions and bug reports are welcome but I make no guarantees that it anything will be added/improved or that bugs will be fixed.  

Please do note that it will cost you money to run it, due to the fees the AI providers charge for use of their APIs.  There is nothing in the code to minimise those costs.  If you run it, you are responsible for all costs incurred.

To use it, copy the `creds.json.sample` file to `creds.json` and fill in the credentials; each one
has a description of what to put into it.  You must provide at least one AI API
key in the `provider_api_keys` dict -- you can see the valid options in the file
providers.json.
