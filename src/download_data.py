# Downloads necessary datasets by opening them in your browser assuming
# you are logged into google since googleAPI is annoying
#
# Also moves them to here, assuming your download directory is at
# ~/Downloads
import webbrowser
import os

categories = ["apple", "banana", "hot dog", "grapes", "donut"]
# Tempaltes that convert categories into whatever they say
url_template = "https://storage.cloud.google.com/quickdraw_dataset/full/simplified/{}.ndjson"
filename_template = "full_simplified_{}.ndjson"

data_directory = os.getcwd() + "/data/download"
download_directory = os.path.expanduser('~/Downloads')

#os.rename('source_file', 'destination_file')

def download_files():
    """Opens all category download links in your computer browser"""
    for category in categories:
        download_url = url_template.format(category)
        webbrowser.open(download_url, new=0, autoraise=True)

def move_files():
    """Move all downloaded files from ~/Downloads to ./data"""
    for category in categories:
        filename = filename_template.format(category)
        os.rename(download_directory + "/" + filename,
                  data_directory + "/" + filename)

if __name__ == "__main__":
    print("Uncomment download_files() to download the files. Uncomment move_files() to move the downloaded files to data directory");
    #download_files()
    #move_files()
