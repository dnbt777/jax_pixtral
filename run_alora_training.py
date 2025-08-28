# train adversarial loras
# rock paper scissors: one lora generates logprobs, another tries to predict them
# 


# ablations
# loss += dkl from standard
# loss += prediction loss from standard
# loss += dkl from opponent lora
# high hide/seek turns
# seek turns >> hide turns
# do this w grpo and do a rollout



# load dataset?
# what data am i going to use for this?
# ideally id like the model to start off 




# init hide_lora


# init seek_lora


# params
hide_turns = 128
seek_turns = 128
epochs = 4

# train

for e in epochs:
    for t_hide in range(hide_turns):
        loss, grads = jax.value_and_grad(train, arg=1)(
                pixtral, hide_lora, seek_lora)
        updates = opt(grads)
        hide_lora += updates

    for t_seek in range(seek_turns):
        loss, grads = jax.value_and_grad(train, arg=1)(
                pixtral, hide_lora, seek_lora)
        updates = opt(grads)
        hide_lora += updates
