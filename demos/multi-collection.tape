# multi-collection.tape — searching across two repos at once with -c repo1,repo2
# Render with: make demos          (or: vhs demos/multi-collection.tape)
# Output → assets/multi-collection.gif

Output assets/multi-collection.gif

Set Theme "Dracula"
Set FontSize 16
Set Width 1200
Set Height 760
Set Padding 20
Set TypingSpeed 50ms
Set Shell zsh

Hide
Type "cd /Users/ugurcanaytar/Desktop/Projects/Recall/recall && rm -rf /tmp/recall-multi && mkdir -p /tmp/recall-multi && export RECALL_DB_PATH=/tmp/recall-multi/index.db && export PATH=$PWD/demos:$PATH && setopt interactive_comments && export PROMPT='$ ' && recall collection add ./examples --name notes >/dev/null && recall collection add ./internal --name src --mask '**/*.go' >/dev/null && recall index >/dev/null && clear" Enter
Sleep 800ms
Show

Type "# two collections indexed: hand-written notes + recall's own Go source" Enter
Sleep 800ms
Type "recall collection list" Enter
Sleep 3500ms

Type "# searching only the notes collection" Enter
Sleep 800ms
Type "recall search 'rate limiter' -c notes -n 3" Enter
Sleep 3500ms

Type "# searching only the source collection" Enter
Sleep 800ms
Type "recall search 'fusion' -c src -n 3" Enter
Sleep 3500ms

Type "# both collections at once with -c notes,src" Enter
Sleep 800ms
Type "recall search 'authentication' -c notes,src -n 4" Enter
Sleep 4000ms
