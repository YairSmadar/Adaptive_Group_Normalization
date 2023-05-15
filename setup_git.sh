#!/bin/bash

# Configure Git
git config --global user.name "YairSmadar"
git config --global user.email "yairsemama@gmail.com"

# Generate SSH key (if it doesn't exist)
if [ ! -f ~/.ssh/id_ed25519 ]; then
  ssh-keygen -t ed25519 -C "yairsemama@gmail.com" -f /home/yair_smadar1/.ssh/id_ed25519 -N ""
fi

# Add the SSH key to the ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519



mv id_ed25519* ~/.ssh/
chmod 600 ~/.ssh/id_ed25519*
