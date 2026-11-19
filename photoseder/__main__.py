"""``python -m photoseder`` opens PhotoSeder in its own desktop window."""

import sys

from photoseder.desktop import main

if __name__ == "__main__":
    sys.exit(main())
