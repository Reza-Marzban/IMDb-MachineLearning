"""
Reza Marzban
"""
import urllib.request
import gzip
import shutil
import os


def dowload_dataset():
    print("Downloading and extracting the IMDB Dataset! Stand By.")
    if not os.path.exists("data"):
        os.makedirs("data")
    # filename_lists = ["name.basics.tsv.gz", "title.akas.tsv.gz", "title.basics.tsv.gz", "title.crew.tsv.gz",
    #                   "title.episode.tsv.gz", "title.principals.tsv.gz", "title.ratings.tsv.gz"]
    filename_lists = ["title.basics.tsv.gz", "title.crew.tsv.gz", "title.ratings.tsv.gz"]
    n = len(filename_lists)
    i = 0
    for fn in filename_lists:
        i += 1
        url = "https://datasets.imdbws.com/"+fn
        urllib.request.urlretrieve(url, "data/"+fn)
        fn_tsv = fn[:-3]
        with gzip.open("data/"+fn, 'rb') as f_in:
            with open("data/"+fn_tsv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove("data/"+fn)
        percent = round(i/n*100)
        print(f"{percent}% complete.")
    print("Done!\n")


if __name__ == "__main__":
    print("\nCS 5783 - Machine Learning\nReza Marzban - A20098444\nFinal Project\n")
    dowload_dataset()