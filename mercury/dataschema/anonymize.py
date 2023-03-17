import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes

import base64
import math


class Anonymize:

    """Cryptographically secure anonymization.

    This class encrypts or hashes lists of strings using cryptographically secure standardized algorithms.
    It can be used with a user defined key or without a key in which case it will produce identical hashes
    across different platforms.

    The key can be given at construction time by setting the environment variable MERCURY_ANONYMIZE_DATASCHEMA_KEY
    or at any later time by calling the .set_key() method.

    Args:
        digest_bits:  This determines the length in (effective) bits of the output hash. As it is encoded in base64,
                      the number of characters will be 1/6 times this number. E.g., 96 (the default) produces 16
                      char long hashes. If this is set to a value other than zero, the output length is fixed, the
                      output is irreversible (cannot be used with .deanonymize_list()) and the algorithm used for
                      hashing is keyed BLAKE2 (https://www.blake2.net/).
                      If this is set to zero, you will get a variable length secure encryption using Galois/Counter
                      Mode AES. (see the argument `safe_crypto`) and the result can be deanonymized with the same key
                      using .deanonymize_list().
        safe_crypto:  This argument selects how the encryption is randomized. If True, the same original text with
                      the same key produces different encrypted texts each time. Note that this will change the
                      cardinality of the set of values to the length of the list.
                      If false (the default) the same text will produce the same output with the same key. This
                      preserves cardinality, but can be a target of attacks when the attacker has access to
                      encoded pairs.
    """

    def __init__(self, digest_bits=96, safe_crypto=False):
        self.digest_bits = digest_bits
        self.safe_crypto = safe_crypto

        plain_key = os.environ.get('MERCURY_ANONYMIZE_DATASCHEMA_KEY')
        plain_key = '<void>' if plain_key is None else plain_key

        hash_key = hashes.Hash(hashes.BLAKE2s(32))

        hash_key.update(plain_key.encode('utf-8'))

        self.hash_key = hash_key.finalize()[0:16]

    def set_key(self, encryption_key):
        """Set the encryption key of an existing `Anonymize` object.

        This changes the encryption key overriding the key possibly defined using the environment variable
        MERCURY_ANONYMIZE_DATASCHEMA_KEY at construction. It can be called any number of times.

        Args:
            encryption_key:  The key as a string.
        """
        hash_key = hashes.Hash(hashes.BLAKE2s(32))

        hash_key.update(encryption_key.encode('utf-8'))

        self.hash_key = hash_key.finalize()[0:16]

    def anonymize_list(self, list_of_str):
        """Anonymize a list of strings.

        This hashes or encrypts a list of strings. The precise function is defined at object construction.
        (See the doc of the class `Anonymize` for details.)

        Args:
            list_of_str:  A list of strings to be anonymized.

        Returns:
            The anonymized list of strings encoded in base64.
        """
        l2 = list()

        if self.digest_bits != 0:
            digest_len = math.ceil(self.digest_bits / 6)

            for s in list_of_str:
                hash = hashes.Hash(hashes.BLAKE2b(64))
                hash.update(self.hash_key)
                hash.update(s.encode('utf-8'))

                l2.append(base64.encodebytes(hash.finalize()).decode()[0:digest_len])
        else:
            aes = AESGCM(self.hash_key)

            if self.safe_crypto:
                for s in list_of_str:
                    nonce = os.urandom(12)		# Must be >8 (min requirement) and multiple of 6 (fixed length in)
                    cipher = aes.encrypt(nonce, s.encode('utf-8'), None)

                    l2.append(base64.encodebytes(nonce + cipher).decode())
            else:
                nonce = b'12345678'
                for s in list_of_str:
                    cipher = aes.encrypt(nonce, s.encode('utf-8'), None)

                    l2.append(base64.encodebytes(cipher).decode())

        return l2

    def anonymize_list_any_type(self, list_of_any):
        """Anonymize a list of anything that supports conversion to string.

        This is a wrapper function over anonymize_list(). It verifies is any element in the list is
        not a string first. If all elements are strings, it passes the list to anonymize_list().
        Otherwise, it creates a new list of string elements and passes that to anonymize_list().

        Args:
            list_of_any:  A list of any data type that supports string conversion via str() to be anonymized.

        Returns:
            The anonymized list of strings encoded in base64.
        """

        assert type(list_of_any) == list

        all_str = True
        for s in list_of_any:
            if type(s) != str:
                all_str = False
                break

        if all_str:
            return self.anonymize_list(list_of_any)

        return self.anonymize_list([str(e) for e in list_of_any])

    def deanonymize_list(self, list_of_str):
        """Deanonymize a list of strings.

        Deanonymizes a list of anonymized strings recovering the original text. This can only be applied if
        the encryption is reversible (The object was created with `digest_bits = 0`) and the key is the same
        key used for encryption.

        Args:
            list_of_str:  A list of strings anonymized using a previous .anonymize_list() call.

        Raises:
            ValueError: When called on an object that does hashing (is created with `digest_bits > 0`)
            rather than encryption.

        Returns:
            The original deanonymized list of strings.
        """
        if self.digest_bits != 0:
            raise ValueError("deanonymize_list() requires passing 'digest_bits = 0' to the constructor.")

        l2 = list()

        aes = AESGCM(self.hash_key)

        if self.safe_crypto:
            for s in list_of_str:
                raw = base64.decodebytes(s.encode())
                nonce = raw[0:12]
                cipher = raw[12:]

                l2.append(aes.decrypt(nonce, cipher, None).decode('utf-8'))
        else:
            nonce = b'12345678'
            for s in list_of_str:
                cipher = base64.decodebytes(s.encode())

                l2.append(aes.decrypt(nonce, cipher, None).decode('utf-8'))

        return l2
