### setup environment
## install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH" # lets us use the uv command immediately
## sync current environment
uv sync



### download pixtral from huggingface
# pixtral requires authentication. you will need an access token.
# once you have been approved to access this repo,
# log in -> go to profile icon (upper right corner) > Access Tokens > Create New Token
# set it to 'read', name it whatever, and create it
# then copy it into the 'password' that this command asks you for:
# https://github.com/git-lfs/git-lfs/discussions/5294#discussioncomment-9556999
HF_HOME="./pixtral"
uv run hf download mistralai/Pixtral-12B-2409 --local-dir ./pixtral --token "$1"



# echo further instructions
echo "done!"
echo "Now run 'uv run {script_name}.py'"
