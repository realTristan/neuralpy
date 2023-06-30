import base64

def base64_encode(s):
    return str(base64.b64encode(s.encode("utf-8")), "utf-8")

def base64_decode(s):
    return str(base64.b64decode(s.encode("utf-8")), "utf-8")