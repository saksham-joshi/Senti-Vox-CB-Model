from typing import List, Final
from scripts.url_email_remover import remove_urls_and_emails

ORD_SPACE = ord(' ')
ORD_NEWLINE = ord('\n')
ORD_TAB = ord('\t')

ENGLISH_A = ord("A")
ENGLISH_a = ord("a")
ENGLISH_Z = ord("Z")
ENGLISH_z = ord("z")
ENGLISH_0 = ord('0')
ENGLISH_9 = ord('9')

HINDI_START = 2304
HINDI_END = 2431
HINDI_0 = 2406
HINDI_9 = 2415

BENGALI_START = 2432
BENGALI_END = 2559
BENGALI_0 = 2534
BENGALI_9 = 2543

GURUMUKHI_START = 2560
GURUMUKHI_END = 2687
GURUMUKHI_0 = 2662
GURUMUKHI_9 = 2671

TAMIL_START = 2944
TAMIL_END = 3071
TAMIL_0 = 3046
TAMIL_9 = 3055

TELUGU_START = 3072
TELUGU_END = 3199
TELUGU_0 = 	3174
TELUGU_9 = 3183

MALAYALAM_START = 3328
MALAYALAM_END = 3455
MALAYALAM_0 = 3430
MALAYALAM_9 = 3439

GUJARATI_START = 2688
GUJARATI_END = 2815
GUJARATI_0 = 2790
GUJARATI_9 = 2799

KANNADA_START = 3200
KANNADA_END = 3327
KANNADA_0 = 3302
KANNADA_9 = 3311

def _isLetterEnglish(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= ENGLISH_A and __order_of_cur_char <= ENGLISH_Z) or (__order_of_cur_char >= ENGLISH_a and __order_of_cur_char <= ENGLISH_z)

def _isLetterHindi(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= HINDI_START and __order_of_cur_char <= HINDI_END)

def _isLetterBengali(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= BENGALI_START and __order_of_cur_char <= BENGALI_END)

def _isLetterGurumukhi(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= GURUMUKHI_START and __order_of_cur_char <= GURUMUKHI_END)

def _isLetterTamil(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= TAMIL_START and __order_of_cur_char <= TAMIL_END)

def _isLetterTelugu(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= TELUGU_START and __order_of_cur_char <= TELUGU_END)

def _isLetterMalayalam(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= MALAYALAM_START and __order_of_cur_char <= MALAYALAM_END)

def _isLetterGujarati(__order_of_cur_char: int) -> bool :
    return (__order_of_cur_char >= GUJARATI_START and __order_of_cur_char <= GUJARATI_END)

def _isLetterKannada(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= KANNADA_START and __order_of_cur_char <= KANNADA_END)

def _isLetterEnglishDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= ENGLISH_0 and __order_of_cur_char <= ENGLISH_9)

def _isLetterHindiDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= HINDI_0 and __order_of_cur_char <= HINDI_9)

def _isLetterBengaliDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= BENGALI_0 and __order_of_cur_char <= BENGALI_9)

def _isLetterGurumukhiDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= GURUMUKHI_0 and __order_of_cur_char <= GURUMUKHI_9)

def _isLetterTamilDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= TAMIL_0 and __order_of_cur_char <= TAMIL_9)

def _isLetterTeluguDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= TELUGU_0 and __order_of_cur_char <= TELUGU_9)

def _isLetterMalayalamDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= MALAYALAM_0 and __order_of_cur_char <= MALAYALAM_9)

def _isLetterGujaratiDigit(__order_of_cur_char : int ) -> bool :
    return (__order_of_cur_char >= GUJARATI_0 and __order_of_cur_char <= GUJARATI_9)

def _isLetterKannadaDigit(__order_of_cur_char : int) -> bool :
    return (__order_of_cur_char >= KANNADA_0 and __order_of_cur_char <= KANNADA_9)

def senti_vox_tokenizer(__text : str) -> List[str] :
    lst = []
    __text = remove_urls_and_emails(__text)
    index = 0
    len : Final = __text.__len__()

    while index < len :

        order_of_cur_char = ord(__text[index])

        script_check_fnc = None

        # ================ DIGITS ================
        if _isLetterEnglishDigit(order_of_cur_char) : script_check_fnc = _isLetterEnglishDigit

        elif _isLetterHindiDigit(order_of_cur_char) : script_check_fnc = _isLetterHindiDigit

        elif _isLetterBengaliDigit(order_of_cur_char) : script_check_fnc = _isLetterBengaliDigit

        elif _isLetterGujaratiDigit(order_of_cur_char) : script_check_fnc = _isLetterGujaratiDigit

        elif _isLetterGurumukhiDigit(order_of_cur_char) : script_check_fnc = _isLetterGurumukhiDigit

        elif _isLetterTamilDigit(order_of_cur_char) : script_check_fnc = _isLetterTamilDigit

        elif _isLetterTeluguDigit(order_of_cur_char) : script_check_fnc = _isLetterTeluguDigit

        elif _isLetterKannadaDigit(order_of_cur_char) : script_check_fnc = _isLetterKannadaDigit

        elif _isLetterMalayalamDigit(order_of_cur_char) : script_check_fnc = _isLetterMalayalamDigit

        # ================= SCRIPTS =============
        elif _isLetterEnglish(order_of_cur_char) : script_check_fnc = _isLetterEnglish

        elif _isLetterHindi(order_of_cur_char)  : script_check_fnc = _isLetterHindi

        elif _isLetterBengali(order_of_cur_char)  : script_check_fnc = _isLetterBengali

        elif _isLetterGujarati(order_of_cur_char)  : script_check_fnc = _isLetterGujarati

        elif _isLetterGurumukhi(order_of_cur_char) : script_check_fnc = _isLetterGurumukhi

        elif _isLetterTamil(order_of_cur_char) : script_check_fnc = _isLetterTamil

        elif _isLetterTelugu(order_of_cur_char) : script_check_fnc = _isLetterTelugu

        elif _isLetterKannada(order_of_cur_char) : script_check_fnc = _isLetterKannada

        elif _isLetterMalayalam(order_of_cur_char) : script_check_fnc = _isLetterMalayalam

        # ============== MISCELLANEOUS =============
        elif order_of_cur_char == ORD_SPACE or order_of_cur_char == ORD_TAB or order_of_cur_char == ORD_NEWLINE:
            index += 1
            continue

        else :
            lst.append(__text[index])
            index += 1
            continue

        start_index = index

        while(index < len and script_check_fnc(ord(__text[index]))) : index += 1

        lst.append(__text[start_index : index])

    return lst