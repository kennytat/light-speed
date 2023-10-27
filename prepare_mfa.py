import regex
import unicodedata
from pathlib import Path
import os

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
    'b', 'c', 'd', 'đ', 'g', 'h', 
    'k', 'l', 'm', 'n', 'p', 'q', 
    'r', 's', 't', 'v', 'x'
]
alphabet = "".join(vietnamese_characters)
space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
num_re = regex.compile(r"([0-9.,]*[0-9])")
keep_text_and_num_re = regex.compile(rf'[^\s{alphabet}.,0-9]')
keep_text_re = regex.compile(rf'[^\s{alphabet}]')

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
    x = unicodedata.normalize('NFKC', x)
    x = x.lower()
    x = num_re.sub(r" \1 ", x)
    x = keep_text_and_num_re.sub(" ", x)
    words = x.split()
    words = [ read_number(w) if num_re.fullmatch(w) else w for w in words ]
    x = " ".join(words)
    x = keep_text_re.sub(" ", x)
    x = space_re.sub(" ", x)
    x = x.strip()
    return x

def create_lexicon():  
  all_text = []
  for fp in sorted(Path("dataset").glob("*.txt")):
      with open(fp, "r", encoding="utf-8") as f:
          text = f.read()
          text = normalize_text(text)
          all_text.append(text)
      # override the text file
      with open(fp, "w", encoding="utf-8") as f:
          f.write(text)
  all_words = sorted(set((" ".join(all_text)).split()))

  with open("lexicon.txt", "w") as f:
      for w in all_words:
          w = w.strip()
          p = list(w)
          p = " ".join(p)
          f.write(f"{w}\t{p}\n")
          
# def merge_lexicon():
#     with open("lexicon.txt", "r", encoding="utf-8") as f:
#       text = f.read().split("\n")
#     with open("lexicon1.txt", "r", encoding="utf-8") as f:
#       text1 = f.read().split("\n")
#     # print(text)
#     text.extend(text1)
#     text = [word.split("\t")[0] for word in text]
#     all_words = sorted(set(text))
#     with open("lexicon-cb.txt", "w") as f:
#       for w in all_words:
#           w = w.strip()
#           p = list(w)
#           p = " ".join(p)
#           f.write(f"{w}\t{p}\n")

def mfa():
  os.system(f"mfa train --num_jobs 40 --use_mp --clean --overwrite --no_textgrid_cleanup --single_speaker --output_format json --output_directory dataset dataset ./lexicon.txt vbx_mfa")
   

if __name__ == "__main__":
  # create_lexicon()
  # merge_lexicon()
  mfa()