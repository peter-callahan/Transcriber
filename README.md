# Transcriber

This is a script that uses genAI to turn screenshots of handwritten notes into text so it can be saved and electronically searched.  The goal is to digitize an analog journal to extract the utility of both analog and digital.

## Usage

Expecting traffic on 5001.

```bash
flask run --host=0.0.0.0 --port=5001
```

## Todo

1. update naming to <DATE> - Title
2. improve multipart uploads, it concatenates messages together after text is extracted. Could make a final API call to combine the 2 texts preserving them as much as possible and blending together.
3. confirm that valid json testing is occurring in the correct place and that I dont have a bunch of extra slop in there.
    a. This is key functionality so include extra logging here to understand exactly what is happening
4. Consider removing responses.json - you really haven't been using this too much. maybe save it but don't serve requests from it?