# This is a SYNHCRONOUS download for bash script
# otherwise bash script would skip over download command
# right after starting it.

import gdown
import sys

url = sys.argv[1]
gdown.download(url, quiet=False, fuzzy=True)
