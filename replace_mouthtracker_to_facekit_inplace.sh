#!/bin/bash

# Move to project root if not already there
cd "$(dirname "$0")"

echo "🔍 Scanning and preparing to replace 'mouthtracker.' → 'facekit.'"

find . -type f -name "*.py" | while read -r file; do
    matches=$(grep -n "mouthtracker\." "$file")
    if [[ -n "$matches" ]]; then
        echo -e "\n📄 File: $file"
        echo "$matches" | while read -r line; do
            lineno=$(echo "$line" | cut -d: -f1)
            content=$(echo "$line" | cut -d: -f2-)
            updated=$(echo "$content" | sed 's/mouthtracker\./facekit./g')
            printf "  - Line %s:\n" "$lineno"
            printf "    OLD: %s\n" "$content"
            printf "    NEW: %s\n" "$updated"
        done
        # Do the replacement (in-place)
        sed -i '' 's/mouthtracker\./facekit./g' "$file"
    fi
done

echo -e "\n✅ All done. Imports updated."
