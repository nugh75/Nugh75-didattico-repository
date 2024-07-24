import chainlit as cl
import litellm

@cl.on_chat_start
def start_chat():
  system_message = {
    "role": "system", 
    "content": "You are a helpful assistant who tries their best to answer questions."
  }
  cl.user_session.set("message_history", [system_message])

@cl.on_message
async def on_message(message: cl.Message):
  messages = cl.user_session.get("message_history")
  if len(message.elements) > 0:
    for element in message.elements:
      with open(element.path, "r") as uploaded_file:
        content = uploaded_file.read()
      messages.append({"role": "user", "content": content})
      confirm_message = cl.Message(content=f"Uploaded file: {element.name}")
      await confirm_message.send()

  msg = cl.Message(content="")
  await msg.send()
  
  messages.append({"role": "user", "content": message.content})

  response = await litellm.acompletion(
    model="ollama/llama3",
    messages = messages,
    api_base="http://localhost:11434",
    stream=True
  )

  async for chunk in response:
    if chunk:
      content = chunk.choices[0].delta.content
      if content:
        await msg.stream_token(content)

  messages.append({"role": "assistant", "content": msg.content})
  await msg.update()