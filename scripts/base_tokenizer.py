class BaseTokenizer :
    class Placeholders :
        URL = "[URL]"
        EMAIL = "[EMAIL]"
        USER = "[USER]"
        NAME = "[NAME]"  # @usernames
        PHONE = "[PHONE]"
        ID = "[ID]"   # Order IDs, tracking numbers, OTPs, promo codes, reference IDs
        IP = "[IP]"
        RT = "[RT]"   # Platform boilerplate: Read more..., <Media omitted>, This message was deleted
