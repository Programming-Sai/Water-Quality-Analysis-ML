#!/bin/bash

# Check if folder name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name> [-g]"
  exit 1
fi

# Set the folder name from the first argument
FOLDER_NAME=$1

# Optionally set flag to add to .gitignore
ADD_TO_GITIGNORE=false

# Check if -g flag is provided to add to .gitignore
if [ "$2" == "-g" ]; then
  ADD_TO_GITIGNORE=true
fi

# Remove the folder from Git tracking but keep it locally
git rm -r --cached "$FOLDER_NAME"

# Commit the changes
git commit -m "Remove $FOLDER_NAME from remote repository"

# Push the changes to the remote repo
git push origin main

# If -g flag is set, add the folder to .gitignore
if [ "$ADD_TO_GITIGNORE" = true ]; then
  echo "$FOLDER_NAME/" >> .gitignore

  # Commit the .gitignore update
  git add .gitignore
  git commit -m "Add $FOLDER_NAME to .gitignore"

  # Push the .gitignore change to the remote repo
  git push origin main
  echo "$FOLDER_NAME removed from remote and added to .gitignore."
else
  echo "$FOLDER_NAME removed from remote but not added to .gitignore."
fi
