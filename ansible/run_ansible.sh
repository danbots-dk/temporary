#!/bin/bash

export ANSIBLE_HOST_KEY_CHECKING=False
ansible-playbook --ask-pass -i ansible_inventory github_pull.yml --extra-vars "ansible_sudo_pass=1606"