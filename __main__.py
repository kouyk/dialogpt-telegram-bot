import os
from functools import wraps

import torch
from telegram import Update, ChatAction
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
from telegram.ext.dispatcher import run_async
from transformers import AutoModelForCausalLM, AutoTokenizer

tg_bot_token = os.environ['TELEGRAM_BOT_TOKEN']  # Bot token given by Telegram Botfather
# tg_admin_id = os.environ['TELEGRAM_ADMIN_ID']  # Telegram admin id, ask @userinfobot (optional)

# Initialise the bot
updater = Updater(tg_bot_token, use_context=True)
dp = updater.dispatcher

# Initialise Dialogpt related entities
print('Loading DialoGPT model...')
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large", )


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(update, context, *args, **kwargs)

        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)


@run_async
def start(update: Update, context: CallbackContext):
    context.chat_data.clear()
    context.chat_data['message_count'] = 0
    update.message.reply_text('[INFO] New conversation started')


dp.add_handler(CommandHandler(['start', 'restart'], start))


@run_async
@send_typing_action
def dialogpt(update: Update, context: CallbackContext):
    # encode the new user input, add the eos_token and return a tensor in PyTorch
    new_user_input_ids = tokenizer.encode(update.message.text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([context.chat_data['chat_history_ids'], new_user_input_ids], dim=-1) \
        if context.chat_data['message_count'] > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    context.chat_data['chat_history_ids'] = model.generate(bot_input_ids, max_length=1000,
                                                           pad_token_id=tokenizer.eos_token_id)

    # decode and reply the user
    update.message.reply_text(tokenizer.decode(context.chat_data['chat_history_ids'][:, bot_input_ids.shape[-1]:][0],
                                               skip_special_tokens=True))
    context.chat_data['message_count'] += 1


# command handlers, e.g. /start

# message handlers, aka all normal text messages
dp.add_handler(MessageHandler(Filters.text, dialogpt))

# Begin the bot
print("Bot running...")
updater.start_polling()
updater.idle()
