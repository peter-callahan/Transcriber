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

## Functionality

Provide an easy way to upload photos of written text from a phone into a UI that allows the user to organize, enhance, and eventually convert the photos into markdown text.

1. The structure of the text and the order defined by the user must be preserved.
2. User defined overrides must apply to the appropriate images (date or location overrides)
3. The text in the photo should not be corrected, touched up, reordered, or changed in any way.
4. Images uploaded together and bundled in the same group represent a single writing event. They should have a single output with concatenates them together while maintaining structure.
5. Uploaded images should be preserved and returned with the markdown conversion of the data so the user can cross reference any points of confusion.
6. Metadata fields should be kept in a distinct portion of the notes and be standardized.