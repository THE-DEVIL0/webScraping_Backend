import os, requests, tempfile

def ensure_local_file(path_or_url: str) -> str:
    """Download remote file if it's a URL, otherwise return local path."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        suffix = os.path.splitext(path_or_url.split("?")[0])[1] or ".jpg"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in response.iter_content(1024):
            tmp_file.write(chunk)
        tmp_file.close()
        return tmp_file.name
    return path_or_url