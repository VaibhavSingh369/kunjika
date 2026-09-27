"""
Devanagari -> Hinglish (romanised Hindi, the way people type it).

Whisper transcribes Hindi speech accurately into Devanagari; this converts the
script only, deterministically, so nothing can be misheard at this stage.

Hindi drops the inherent "a" (schwa) in speech: करता is "karta", not "karata".
Standard rule, applied right to left: delete a schwa that sits between a
vowel+consonant and a consonant+vowel (VC_CV), and at the end of a word.
"""

import difflib
import re

CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
    "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
    "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh",
    "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
    "ह": "h", "ळ": "l",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y",
}
NUKTA_OF = {"क": "q", "ख": "kh", "ग": "gh", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
          "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
MATRAS = {"ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e",
          "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e"}
VIRAMA, NUKTA, ANUSVARA, CHANDRA, VISARGA = "्", "़", "ं", "ँ", "ः"

# English words Whisper writes in Devanagari when they are spoken inside Hindi
LOANWORDS = {
    "लाइट": "light", "लाईट": "light", "लाइटें": "lights", "ऑन": "on", "ओन": "on",
    "ऑफ": "off", "ऑफ़": "off", "ओफ": "off", "फैन": "fan", "गेट": "gate", "डोर": "door",
    "टीवी": "TV", "एसी": "AC", "टाइम": "time", "टाईम": "time", "म्यूजिक": "music",
    "म्यूज़िक": "music", "सॉन्ग": "song", "प्लीज": "please", "प्लीज़": "please", "ओके": "OK",
    "बल्ब": "bulb", "रूम": "room", "फोन": "phone", "फ़ोन": "phone", "वॉल्यूम": "volume",
    "टेम्परेचर": "temperature", "मोड": "mode", "अलार्म": "alarm", "सेट": "set",
    "स्टार्ट": "start", "स्टॉप": "stop", "कंप्यूटर": "computer", "लैपटॉप": "laptop",
    "मोबाइल": "mobile", "नंबर": "number", "सिस्टम": "system", "रिपोर्ट": "report",
    "स्टेटस": "status", "चेक": "check", "एप्पल": "apple", "सैटेलाइट": "satellite",
    "फ़ैन": "fan", "स्पीड": "speed", "डिग्री": "degree", "किचन": "kitchen", "बेडरूम": "bedroom", "मिनट": "minute",
}
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _units(word):
    """[roman consonant(s), vowel, coda, independent-vowel?]; vowel is None for
    an inherent schwa and "" after a virama."""
    units, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONS:
            roman = CONS[ch]
            i += 1
            if i < n and word[i] == NUKTA:
                roman = NUKTA_OF.get(ch, roman)
                i += 1
            vowel = None
            if i < n and word[i] in MATRAS:
                vowel = MATRAS[word[i]]
                i += 1
            elif i < n and word[i] == VIRAMA:
                vowel = ""
                i += 1
            coda = ""
            while i < n and word[i] in (ANUSVARA, CHANDRA, VISARGA):
                coda += word[i]
                i += 1
            units.append([roman, vowel, coda, False])
        elif ch in VOWELS:
            i += 1
            coda = ""
            while i < n and word[i] in (ANUSVARA, CHANDRA, VISARGA):
                coda += word[i]
                i += 1
            units.append(["", VOWELS[ch], coda, True])
        else:
            i += 1                                  # stray marks: ignore
    return units


def _has_vowel(u):
    return u[1] is None or u[1] != ""


def _word(word):
    if word in LOANWORDS:
        return LOANWORDS[word]
    u = _units(word)
    if not u:
        return word
    # schwa deletion: the final one, then VC_CV scanning right to left
    if len(u) > 1 and u[-1][1] is None and not u[-1][2]:
        u[-1][1] = ""
    for k in range(len(u) - 2, 0, -1):
        if u[k][1] is None and not u[k][2]:
            nxt = u[k + 1]
            if _has_vowel(u[k - 1]) and not nxt[3] and _has_vowel(nxt):
                u[k][1] = ""
    n_vowels = sum(1 for x in u if _has_vowel(x))
    out = []
    for k, (c, v, coda, indep) in enumerate(u):
        v = "a" if v is None else v
        last = k == len(u) - 1
        if not indep and n_vowels == 1 and c and MATRA_A(word) and v == "a" and not last:
            v = "aa"                                # one-syllable words: baat, naam, kaam
        if v in ("ee", "oo") and last:
            v = {"ee": "i", "oo": "u"}[v]           # word-final: pani, rahi, nahin, tu
        nasal = ""
        for m in coda:
            if m == VISARGA:
                nasal += "h"
            else:
                nxt = u[k + 1][0] if k + 1 < len(u) else ""
                nasal += "m" if m == ANUSVARA and nxt[:1] in ("p", "b", "m") else "n"
        out.append(c + v + nasal)
    return "".join(out)


def MATRA_A(word):
    return "ा" in word


def to_hinglish(text):
    text = text.translate(DIGITS).replace("।", ".").replace("॥", ".")
    return re.sub(r"[\u0900-\u097F]+", lambda m: _word(m.group(0)), text)


def normalise_keyword(text, keyword="Kunjika"):
    """The board already detected the keyword, so a near-miss spelling of the
    first word (kunjiga, kuijika) is shown as the keyword."""
    m = re.match(r"(\s*)([A-Za-z]+)(.*)", text, re.S)
    if m and difflib.SequenceMatcher(None, m.group(2).lower(), keyword.lower()).ratio() >= 0.6:
        return m.group(1) + keyword + m.group(3)
    return text
