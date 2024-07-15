import os
import regex
import re
import json
import unicodedata
import shutil
from datetime import date
from pathlib import Path
from viphoneme import vi2IPA, vi2IPA_split, syms
from num2words import num2words
from icu import Collator, Locale
from langdetect import detect

with open("./config.json", "rb") as f:
    config = json.load(f)
    
dataset_dir = config["data"]["dataset_dir"] # Path to .wavs and .txt directory
output_dir = config["data"]["output_dir"] # Path to .wavs and .txt directory
ckpt_dir = os.path.join(output_dir, "ckpts")
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
shutil.copyfile("config.json", f"{ckpt_dir}/config.json")
lexicon_file = f"{ckpt_dir}/lexicon.txt"

vietnamese_characters = [
    'a', 'à', 'á', 'ả', 'ã', 'ạ',
    'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
    'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
    'e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
    'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',
    'i', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
    'o', 'ò', 'ó', 'ỏ', 'õ', 'ọ',
    'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
    'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
    'u', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
    'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',
    'y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',
    'b', 'c', 'd', 'đ', 'f', 'g', 
    'h', 'j', 'k', 'l', 'm', 'n',
    'p', 'q', 'r', 's', 't', 'v',
    'w', 'x', 'z'
]
alphabet = "".join(vietnamese_characters)
space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
num_re = regex.compile(r"([0-9.,]*[0-9])")
keep_text_and_num_re = regex.compile(rf'[^\s{alphabet}.,0-9]')
keep_text_re = regex.compile(rf'[^\s{alphabet}]')
collator = Collator.createInstance(Locale('vi_VN.UTF-8'))

def read_number(num: str) -> str:
    if len(num) == 1:
        return digits[int(num)]
    elif len(num) == 2 and num.isdigit():
        n = int(num)
        end = digits[n % 10]
        if n == 10:
            return "mười"
        if n % 10 == 5:
            end = "lăm"
        if n % 10 == 0:
            return digits[n // 10] + " mươi"
        elif n < 20:
            return "mười " + end
        else:
            if n % 10 == 1:
                end = "mốt"
            return digits[n // 10] + " mươi " + end
    elif len(num) == 3 and num.isdigit():
        n = int(num)
        if n % 100 == 0:
            return digits[n // 100] + " trăm"
        elif num[1] == "0":
            return digits[n // 100] + " trăm lẻ " + digits[n%100]
        else:
            return digits[n // 100] + " trăm " + read_number(num[1:])
    elif "," in num:
        n1, n2 = num.split(",")
        return read_number(n1) + " phẩy " + read_number(n2)
    elif "." in num:
        parts = num.split(".")
        if len(parts) == 2:
            if parts[1] == "000":
                return read_number(parts[0]) + " ngàn"
            elif parts[1].startswith("00"):
                end = digits[int(parts[1][2:])]
                return read_number(parts[0]) + " ngàn lẻ " + end
            else:
                return read_number(parts[0]) + " ngàn " + read_number(parts[1])
        elif len(parts) == 3:
            if num == "1.000.000":
                return "một triệu"
            elif num == "1.100.000":
                return "một triệu một trăm ngàn"
            elif num == "1.820.000":
                return "một triệu tám trăm hai mươi ngàn"
            elif num == "3.640.000":
                return "ba triệu sáu trăm bốn mươi ngàn"
            else:
                raise ValueError(f"Cannot convert {num}")
    else:
        raise ValueError(f"Cannot convert {num}")
    return num

def normalize_text(x):
    try:
      lang = detect(x)
    except:
      lang = "en"
    print(f"normalize_text: {lang} | {x}")
    x = unicodedata.normalize('NFKC', x)
    x = x.lower()
    x = num_re.sub(r" \1 ", x)
    x = keep_text_and_num_re.sub(" ", x)
    words = x.split()
    words = [ read_number(w) if num_re.fullmatch(w) else w for w in words ] if lang == "vi" else [ num2words(w.replace(",","")) if num_re.fullmatch(w) else w for w in words ]
    x = " ".join(words)
    x = keep_text_re.sub(" ", x)
    x = space_re.sub(" ", x)
    x = x.strip()
    return x

def create_lexicon():
  all_text = []
  with open("fulltext.txt", "w") as ft:
    for fp in sorted(Path(dataset_dir).glob("*.txt")):
      with open(fp, "r", encoding="utf-8") as f:
          text = f.read()
          print("text:", text)
          if text != "" and text != "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn":
            ft.write(f"{fp}|{text}\n")
            text = normalize_text(text)
            all_text.append(text)
          else:
            os.system(f"rm {fp}")
      # override the text file
      with open(fp, "w", encoding="utf-8") as f:
          f.write(text)
  all_words = sorted(set((" ".join(all_text)).split()), key=collator.getSortKey)

  with open(lexicon_file, "w") as f:
      for w in all_words:
          w = w.strip()
          p = list(w)
          p = " ".join(p)
          f.write(f"{w}\t{p}\n")
          
def merge_lexicon():
    with open(lexicon_file, "r", encoding="utf-8") as f:
      text = f.read().split("\n")
    with open(f"base_lexicon.txt", "r", encoding="utf-8") as f:
      text1 = f.read().split("\n")
    # print(text)
    text.extend(text1)
    words = [word.split("\t")[0] for word in text]
    all_words = sorted(set(words), key=collator.getSortKey)
    with open("lexicon.txt", "w") as f:
      for w in all_words:
        found = [element for element in text if w in element]
        f.write(f"{found[0]}\n")


def mfa(name):
  os.system(f"mfa train --num_jobs 10 --use_mp --clean --overwrite --no_textgrid_cleanup --single_speaker --output_format json --output_directory {dataset_dir} {dataset_dir} {lexicon_file} mfa_{name}")
  
if __name__ == "__main__":
  create_lexicon()
  merge_lexicon()
  model_name = f"vtts-{date.today()}"
  mfa(model_name)