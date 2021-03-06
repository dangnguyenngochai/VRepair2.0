/*
 * Copyright (c) 2006-2012 Red Hat, Inc.
 *
 * All rights reserved.
 *
 * Author: Steven Dake (sdake@redhat.com)
 *         Christine Caulfield (ccaulfie@redhat.com)
 *         Jan Friesse (jfriesse@redhat.com)
 *         Fabio M. Di Nitto (fdinitto@redhat.com)
 *
 * This software licensed under BSD license, the text of which follows:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the MontaVista Software, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <nss.h>
#include <pk11pub.h>
#include <pkcs11.h>
#include <prerror.h>
#include <blapit.h>
#include <hasht.h>

#define LOGSYS_UTILS_ONLY 1
#include <corosync/logsys.h>
#include <corosync/totem/totem.h>
#include "totemcrypto.h"

/*
 * define onwire crypto header
 */

struct crypto_config_header {
	uint8_t	crypto_cipher_type;
	uint8_t	crypto_hash_type;
	uint8_t __pad0;
	uint8_t __pad1;
} __attribute__((packed));

/*
 * crypto definitions and conversion tables
 */

#define SALT_SIZE 16

/*
 * This are defined in new NSS. For older one, we will define our own
 */
#ifndef AES_256_KEY_LENGTH
#define AES_256_KEY_LENGTH 32
#endif

#ifndef AES_192_KEY_LENGTH
#define AES_192_KEY_LENGTH 24
#endif

#ifndef AES_128_KEY_LENGTH
#define AES_128_KEY_LENGTH 16
#endif

/*
 * while CRYPTO_CIPHER_TYPE_2_2 is not a real cipher at all,
 * we still allocate a value for it because we use crypto_crypt_t
 * internally and we don't want overlaps
 */

enum crypto_crypt_t {
	CRYPTO_CIPHER_TYPE_NONE = 0,
	CRYPTO_CIPHER_TYPE_AES256 = 1,
	CRYPTO_CIPHER_TYPE_AES192 = 2,
	CRYPTO_CIPHER_TYPE_AES128 = 3,
	CRYPTO_CIPHER_TYPE_3DES = 4,
	CRYPTO_CIPHER_TYPE_2_2 = UINT8_MAX
};

CK_MECHANISM_TYPE cipher_to_nss[] = {
	0,				/* CRYPTO_CIPHER_TYPE_NONE */
	CKM_AES_CBC_PAD,		/* CRYPTO_CIPHER_TYPE_AES256 */
	CKM_AES_CBC_PAD,		/* CRYPTO_CIPHER_TYPE_AES192 */
	CKM_AES_CBC_PAD,		/* CRYPTO_CIPHER_TYPE_AES128 */
	CKM_DES3_CBC_PAD		/* CRYPTO_CIPHER_TYPE_3DES */
};

size_t cipher_key_len[] = {
	0,				/* CRYPTO_CIPHER_TYPE_NONE */
	AES_256_KEY_LENGTH,		/* CRYPTO_CIPHER_TYPE_AES256 */
	AES_192_KEY_LENGTH,		/* CRYPTO_CIPHER_TYPE_AES192 */
	AES_128_KEY_LENGTH,		/* CRYPTO_CIPHER_TYPE_AES128 */
	24				/* CRYPTO_CIPHER_TYPE_3DES - no magic in nss headers */
};

size_t cypher_block_len[] = {
	0,				/* CRYPTO_CIPHER_TYPE_NONE */
	AES_BLOCK_SIZE,			/* CRYPTO_CIPHER_TYPE_AES256 */
	AES_BLOCK_SIZE,			/* CRYPTO_CIPHER_TYPE_AES192 */
	AES_BLOCK_SIZE,			/* CRYPTO_CIPHER_TYPE_AES128 */
	0				/* CRYPTO_CIPHER_TYPE_3DES */
};

/*
 * hash definitions and conversion tables
 */

/*
 * while CRYPTO_HASH_TYPE_2_2 is not a real hash mechanism at all,
 * we still allocate a value for it because we use crypto_hash_t
 * internally and we don't want overlaps
 */

enum crypto_hash_t {
	CRYPTO_HASH_TYPE_NONE	= 0,
	CRYPTO_HASH_TYPE_MD5	= 1,
	CRYPTO_HASH_TYPE_SHA1	= 2,
	CRYPTO_HASH_TYPE_SHA256	= 3,
	CRYPTO_HASH_TYPE_SHA384	= 4,
	CRYPTO_HASH_TYPE_SHA512	= 5,
	CRYPTO_HASH_TYPE_2_2	= UINT8_MAX
};

CK_MECHANISM_TYPE hash_to_nss[] = {
	0,				/* CRYPTO_HASH_TYPE_NONE */
	CKM_MD5_HMAC,			/* CRYPTO_HASH_TYPE_MD5 */
	CKM_SHA_1_HMAC,			/* CRYPTO_HASH_TYPE_SHA1 */
	CKM_SHA256_HMAC,		/* CRYPTO_HASH_TYPE_SHA256 */
	CKM_SHA384_HMAC,		/* CRYPTO_HASH_TYPE_SHA384 */
	CKM_SHA512_HMAC			/* CRYPTO_HASH_TYPE_SHA512 */
};

size_t hash_len[] = {
	0,				/* CRYPTO_HASH_TYPE_NONE */
	MD5_LENGTH,			/* CRYPTO_HASH_TYPE_MD5 */
	SHA1_LENGTH,			/* CRYPTO_HASH_TYPE_SHA1 */
	SHA256_LENGTH,			/* CRYPTO_HASH_TYPE_SHA256 */
	SHA384_LENGTH,			/* CRYPTO_HASH_TYPE_SHA384 */
	SHA512_LENGTH			/* CRYPTO_HASH_TYPE_SHA512 */
};

size_t hash_block_len[] = {
	0,				/* CRYPTO_HASH_TYPE_NONE */
	MD5_BLOCK_LENGTH,		/* CRYPTO_HASH_TYPE_MD5 */
	SHA1_BLOCK_LENGTH,		/* CRYPTO_HASH_TYPE_SHA1 */
	SHA256_BLOCK_LENGTH,		/* CRYPTO_HASH_TYPE_SHA256 */
	SHA384_BLOCK_LENGTH,		/* CRYPTO_HASH_TYPE_SHA384 */
	SHA512_BLOCK_LENGTH		/* CRYPTO_HASH_TYPE_SHA512 */
};

/*
 * crypto on-wire compat
 */

enum crypto_compat_t {
	CRYPTO_COMPAT_2_0	= 0,
	CRYPTO_COMPAT_2_2	= 1
};

struct crypto_instance {
	PK11SymKey   *nss_sym_key;
	PK11SymKey   *nss_sym_key_sign;

	unsigned char private_key[1024];

	unsigned int private_key_len;

	enum crypto_crypt_t crypto_cipher_type;

	enum crypto_hash_t crypto_hash_type;

	enum crypto_compat_t crypto_compat_type;

	unsigned int crypto_header_size;

	void (*log_printf_func) (
		int level,
		int subsys,
		const char *function,
		const char *file,
		int line,
		const char *format,
		...)__attribute__((format(printf, 6, 7)));

	int log_level_security;
	int log_level_notice;
	int log_level_error;
	int log_subsys_id;
};

#define log_printf(level, format, args...)				\
do {									\
	instance->log_printf_func (					\
		level, instance->log_subsys_id,				\
		__FUNCTION__, __FILE__, __LINE__,			\
		(const char *)format, ##args);				\
} while (0);

/*
 * compat functions
 */

static int string_to_crypto_compat_type(const char* crypto_compat_type)
{
	if (strcmp(crypto_compat_type, "2.0") == 0) {
		return CRYPTO_COMPAT_2_0;
	} else if (strcmp(crypto_compat_type, "2.1") == 0) {
		return CRYPTO_COMPAT_2_2;
	}
	return CRYPTO_COMPAT_2_2;
}

/*
 * crypt/decrypt functions
 */

static int string_to_crypto_cipher_type(const char* crypto_cipher_type)
{
	if (strcmp(crypto_cipher_type, "none") == 0) {
		return CRYPTO_CIPHER_TYPE_NONE;
	} else if (strcmp(crypto_cipher_type, "aes256") == 0) {
		return CRYPTO_CIPHER_TYPE_AES256;
	} else if (strcmp(crypto_cipher_type, "aes192") == 0) {
		return CRYPTO_CIPHER_TYPE_AES192;
	} else if (strcmp(crypto_cipher_type, "aes128") == 0) {
		return CRYPTO_CIPHER_TYPE_AES128;
	} else if (strcmp(crypto_cipher_type, "3des") == 0) {
		return CRYPTO_CIPHER_TYPE_3DES;
	}
	return CRYPTO_CIPHER_TYPE_AES256;
}

static int init_nss_crypto(struct crypto_instance *instance)
{
	PK11SlotInfo*	crypt_slot = NULL;
	SECItem		crypt_param;

	if (!cipher_to_nss[instance->crypto_cipher_type]) {
		return 0;
	}

	crypt_param.type = siBuffer;
	crypt_param.data = instance->private_key;
	crypt_param.len = cipher_key_len[instance->crypto_cipher_type];

	crypt_slot = PK11_GetBestSlot(cipher_to_nss[instance->crypto_cipher_type], NULL);
	if (crypt_slot == NULL) {
		log_printf(instance->log_level_security, "Unable to find security slot (err %d)",
			   PR_GetError());
		return -1;
	}

	instance->nss_sym_key = PK11_ImportSymKey(crypt_slot,
						  cipher_to_nss[instance->crypto_cipher_type],
						  PK11_OriginUnwrap, CKA_ENCRYPT|CKA_DECRYPT,
						  &crypt_param, NULL);
	if (instance->nss_sym_key == NULL) {
		log_printf(instance->log_level_security, "Failure to import key into NSS (err %d)",
			   PR_GetError());
		return -1;
	}

	PK11_FreeSlot(crypt_slot);

	return 0;
}

static int encrypt_nss(
	struct crypto_instance *instance,
	const unsigned char *buf_in,
	const size_t buf_in_len,
	unsigned char *buf_out,
	size_t *buf_out_len)
{
	PK11Context*	crypt_context = NULL;
	SECItem		crypt_param;
	SECItem		*nss_sec_param = NULL;
	int		tmp1_outlen = 0;
	unsigned int	tmp2_outlen = 0;
	unsigned char	*salt = buf_out;
	unsigned char	*data = buf_out + SALT_SIZE;
	int		err = -1;

	if (!cipher_to_nss[instance->crypto_cipher_type]) {
		memcpy(buf_out, buf_in, buf_in_len);
		*buf_out_len = buf_in_len;
		return 0;
	}

	if (PK11_GenerateRandom (salt, SALT_SIZE) != SECSuccess) {
		log_printf(instance->log_level_security,
			"Failure to generate a random number %d",
			PR_GetError());
		goto out;
	}

	crypt_param.type = siBuffer;
	crypt_param.data = salt;
	crypt_param.len = SALT_SIZE;

	nss_sec_param = PK11_ParamFromIV (cipher_to_nss[instance->crypto_cipher_type],
					  &crypt_param);
	if (nss_sec_param == NULL) {
		log_printf(instance->log_level_security,
			   "Failure to set up PKCS11 param (err %d)",
			   PR_GetError());
		goto out;
	}

	/*
	 * Create cipher context for encryption
	 */
	crypt_context = PK11_CreateContextBySymKey (cipher_to_nss[instance->crypto_cipher_type],
						    CKA_ENCRYPT,
						    instance->nss_sym_key,
						    nss_sec_param);
	if (!crypt_context) {
		log_printf(instance->log_level_security,
			   "PK11_CreateContext failed (encrypt) crypt_type=%d (err %d)",
			   (int)cipher_to_nss[instance->crypto_cipher_type],
			   PR_GetError());
		goto out;
	}

	if (PK11_CipherOp(crypt_context, data,
			  &tmp1_outlen,
			  FRAME_SIZE_MAX - instance->crypto_header_size,
			  (unsigned char *)buf_in, buf_in_len) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_CipherOp failed (encrypt) crypt_type=%d (err %d)",
			   (int)cipher_to_nss[instance->crypto_cipher_type],
			   PR_GetError());
		goto out;
	}

	if (PK11_DigestFinal(crypt_context, data + tmp1_outlen,
			     &tmp2_outlen, FRAME_SIZE_MAX - tmp1_outlen) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_DigestFinal failed (encrypt) crypt_type=%d (err %d)",
			   (int)cipher_to_nss[instance->crypto_cipher_type],
			   PR_GetError());
		goto out;

	}

	*buf_out_len = tmp1_outlen + tmp2_outlen + SALT_SIZE;

	err = 0;

out:
	if (crypt_context) {
		PK11_DestroyContext(crypt_context, PR_TRUE);
	}
	if (nss_sec_param) {
		SECITEM_FreeItem(nss_sec_param, PR_TRUE);
	}
	return err;
}

static int decrypt_nss (
	struct crypto_instance *instance,
	unsigned char *buf,
	int *buf_len)
{
	PK11Context*	decrypt_context = NULL;
	SECItem		decrypt_param;
	int		tmp1_outlen = 0;
	unsigned int	tmp2_outlen = 0;
	unsigned char	*salt = buf;
	unsigned char	*data = salt + SALT_SIZE;
	int		datalen = *buf_len - SALT_SIZE;
	unsigned char	outbuf[FRAME_SIZE_MAX];
	int		outbuf_len;
	int		err = -1;

	if (!cipher_to_nss[instance->crypto_cipher_type]) {
		return 0;
	}

	/* Create cipher context for decryption */
	decrypt_param.type = siBuffer;
	decrypt_param.data = salt;
	decrypt_param.len = SALT_SIZE;

	decrypt_context = PK11_CreateContextBySymKey(cipher_to_nss[instance->crypto_cipher_type],
						     CKA_DECRYPT,
						     instance->nss_sym_key, &decrypt_param);
	if (!decrypt_context) {
		log_printf(instance->log_level_security,
			   "PK11_CreateContext (decrypt) failed (err %d)",
			   PR_GetError());
		goto out;
	}

	if (PK11_CipherOp(decrypt_context, outbuf, &tmp1_outlen,
			  sizeof(outbuf), data, datalen) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_CipherOp (decrypt) failed (err %d)",
			   PR_GetError());
		goto out;
	}

	if (PK11_DigestFinal(decrypt_context, outbuf + tmp1_outlen, &tmp2_outlen,
			     sizeof(outbuf) - tmp1_outlen) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_DigestFinal (decrypt) failed (err %d)",
			   PR_GetError()); 
		goto out;
	}

	outbuf_len = tmp1_outlen + tmp2_outlen;

	memset(buf, 0, *buf_len);
	memcpy(buf, outbuf, outbuf_len);

	*buf_len = outbuf_len;

	err = 0;

out:
	if (decrypt_context) {
		PK11_DestroyContext(decrypt_context, PR_TRUE);
	}

	return err;
}


/*
 * hash/hmac/digest functions
 */

static int string_to_crypto_hash_type(const char* crypto_hash_type)
{
	if (strcmp(crypto_hash_type, "none") == 0) {
		return CRYPTO_HASH_TYPE_NONE;
	} else if (strcmp(crypto_hash_type, "md5") == 0) {
		return CRYPTO_HASH_TYPE_MD5;
	} else if (strcmp(crypto_hash_type, "sha1") == 0) {
		return CRYPTO_HASH_TYPE_SHA1;
	} else if (strcmp(crypto_hash_type, "sha256") == 0) {
		return CRYPTO_HASH_TYPE_SHA256;
	} else if (strcmp(crypto_hash_type, "sha384") == 0) {
		return CRYPTO_HASH_TYPE_SHA384;
	} else if (strcmp(crypto_hash_type, "sha512") == 0) {
		return CRYPTO_HASH_TYPE_SHA512;
	}

	return CRYPTO_HASH_TYPE_SHA1;
}

static int init_nss_hash(struct crypto_instance *instance)
{
	PK11SlotInfo*	hash_slot = NULL;
	SECItem		hash_param;

	if (!hash_to_nss[instance->crypto_hash_type]) {
		return 0;
	}

	hash_param.type = siBuffer;
	hash_param.data = 0;
	hash_param.len = 0;

	hash_slot = PK11_GetBestSlot(hash_to_nss[instance->crypto_hash_type], NULL);
	if (hash_slot == NULL) {
		log_printf(instance->log_level_security, "Unable to find security slot (err %d)",
			   PR_GetError());
		return -1;
	}

	instance->nss_sym_key_sign = PK11_ImportSymKey(hash_slot,
						       hash_to_nss[instance->crypto_hash_type],
						       PK11_OriginUnwrap, CKA_SIGN,
						       &hash_param, NULL);
	if (instance->nss_sym_key_sign == NULL) {
		log_printf(instance->log_level_security, "Failure to import key into NSS (err %d)",
			   PR_GetError());
		return -1;
	}

	PK11_FreeSlot(hash_slot);

	return 0;
}

static int calculate_nss_hash(
	struct crypto_instance *instance,
	const unsigned char *buf,
	const size_t buf_len,
	unsigned char *hash)
{
	PK11Context*	hash_context = NULL;
	SECItem		hash_param;
	unsigned int	hash_tmp_outlen = 0;
	unsigned char	hash_block[hash_block_len[instance->crypto_hash_type]];
	int		err = -1;

	/* Now do the digest */
	hash_param.type = siBuffer;
	hash_param.data = 0;
	hash_param.len = 0;

	hash_context = PK11_CreateContextBySymKey(hash_to_nss[instance->crypto_hash_type],
						 CKA_SIGN,
						 instance->nss_sym_key_sign,
						 &hash_param);

	if (!hash_context) {
		log_printf(instance->log_level_security,
			   "PK11_CreateContext failed (hash) hash_type=%d (err %d)",
			   (int)hash_to_nss[instance->crypto_hash_type],
			   PR_GetError());
		goto out;
	}

	if (PK11_DigestBegin(hash_context) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_DigestBegin failed (hash) hash_type=%d (err %d)",
			   (int)hash_to_nss[instance->crypto_hash_type],
			   PR_GetError());
		goto out;
	}

	if (PK11_DigestOp(hash_context,
			  buf,
			  buf_len) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_DigestOp failed (hash) hash_type=%d (err %d)",
			   (int)hash_to_nss[instance->crypto_hash_type],
			   PR_GetError());
		goto out;
	}

	if (PK11_DigestFinal(hash_context,
			     hash_block,
			     &hash_tmp_outlen,
			     hash_block_len[instance->crypto_hash_type]) != SECSuccess) {
		log_printf(instance->log_level_security,
			   "PK11_DigestFinale failed (hash) hash_type=%d (err %d)",
			   (int)hash_to_nss[instance->crypto_hash_type],
			   PR_GetError());
		goto out;
	}

	memcpy(hash, hash_block, hash_len[instance->crypto_hash_type]);
	err = 0;

out:
	if (hash_context) {
		PK11_DestroyContext(hash_context, PR_TRUE);
	}

	return err;
}

/*
 * global/glue nss functions
 */

static int init_nss_db(struct crypto_instance *instance)
{
	if ((!cipher_to_nss[instance->crypto_cipher_type]) &&
	    (!hash_to_nss[instance->crypto_hash_type])) {
		return 0;
	}

	if (NSS_NoDB_Init(".") != SECSuccess) {
		log_printf(instance->log_level_security, "NSS DB initialization failed (err %d)",
			   PR_GetError());
		return -1;
	}

	return 0;
}

static int init_nss(struct crypto_instance *instance,
		    const char *crypto_cipher_type,
		    const char *crypto_hash_type,
		    const char *crypto_compat_type)
{
	log_printf(instance->log_level_notice,
		   "Initializing transmit/receive security (NSS) crypto: %s hash: %s compat: %s",
		   crypto_cipher_type, crypto_hash_type, crypto_compat_type);

	if (init_nss_db(instance) < 0) {
		return -1;
	}

	if (init_nss_crypto(instance) < 0) {
		return -1;
	}

	if (init_nss_hash(instance) < 0) {
		return -1;
	}

	return 0;
}

static int encrypt_and_sign_nss_2_0 (
	struct crypto_instance *instance,
	const unsigned char *buf_in,
	const size_t buf_in_len,
	unsigned char *buf_out,
	size_t *buf_out_len)
{
	unsigned char	*hash = buf_out;
	unsigned char	*data = hash + hash_len[instance->crypto_hash_type];

	if (encrypt_nss(instance, buf_in, buf_in_len, data, buf_out_len) < 0) {
		return -1;
	}

	if (hash_to_nss[instance->crypto_hash_type]) {
		if (calculate_nss_hash(instance, data, *buf_out_len, hash) < 0) {
			return -1;
		}
		*buf_out_len = *buf_out_len + hash_len[instance->crypto_hash_type];
	}

	return 0;
}

static int encrypt_and_sign_nss_2_2 (
	struct crypto_instance *instance,
	const unsigned char *buf_in,
	const size_t buf_in_len,
	unsigned char *buf_out,
	size_t *buf_out_len)
{
	if (encrypt_nss(instance,
			buf_in, buf_in_len,
			buf_out + sizeof(struct crypto_config_header), buf_out_len) < 0) {
		return -1;
	}

	*buf_out_len += sizeof(struct crypto_config_header);

	if (hash_to_nss[instance->crypto_hash_type]) {
		if (calculate_nss_hash(instance, buf_out, *buf_out_len, buf_out + *buf_out_len) < 0) {
			return -1;
		}
		*buf_out_len += hash_len[instance->crypto_hash_type];
	}

	return 0;
}

static int authenticate_and_decrypt_nss_2_0 (
	struct crypto_instance *instance,
	unsigned char *buf,
	int *buf_len)
{
	if (hash_to_nss[instance->crypto_hash_type]) {
		unsigned char	tmp_hash[hash_len[instance->crypto_hash_type]];
		unsigned char	*hash = buf;
		unsigned char	*data = hash + hash_len[instance->crypto_hash_type];
		int		datalen = *buf_len - hash_len[instance->crypto_hash_type];

		if (calculate_nss_hash(instance, data, datalen, tmp_hash) < 0) {
			return -1;
		}

		if (memcmp(tmp_hash, hash, hash_len[instance->crypto_hash_type]) != 0) {
			log_printf(instance->log_level_error, "Digest does not match");
			return -1;
		}

		memmove(buf, data, datalen);
		*buf_len = datalen;
	}

	if (decrypt_nss(instance, buf, buf_len) < 0) {
		return -1;
	}

	return 0;
}

static int authenticate_nss_2_2 (
	struct crypto_instance *instance,
	unsigned char *buf,
	int *buf_len)
{
	if (hash_to_nss[instance->crypto_hash_type]) {
		unsigned char	tmp_hash[hash_len[instance->crypto_hash_type]];
		int             datalen = *buf_len - hash_len[instance->crypto_hash_type];

		if (calculate_nss_hash(instance, buf, datalen, tmp_hash) < 0) {
			return -1;
		}

		if (memcmp(tmp_hash, buf + datalen, hash_len[instance->crypto_hash_type]) != 0) {
			log_printf(instance->log_level_error, "Digest does not match");
			return -1;
		}
		*buf_len = datalen;
	}

	return 0;
}

static int decrypt_nss_2_2 (
	struct crypto_instance *instance,
	unsigned char *buf,
	int *buf_len)
{
	*buf_len -= sizeof(struct crypto_config_header);

	if (decrypt_nss(instance, buf + sizeof(struct crypto_config_header), buf_len) < 0) {
		return -1;
	}

	return 0;
}

/*
 * exported API
 */

size_t crypto_sec_header_size(
	const char *crypto_cipher_type,
	const char *crypto_hash_type)
{
	int crypto_cipher = string_to_crypto_cipher_type(crypto_cipher_type);
	int crypto_hash = string_to_crypto_hash_type(crypto_hash_type);
	size_t hdr_size = 0;

	hdr_size = sizeof(struct crypto_config_header);

	if (crypto_hash) {
		hdr_size += hash_len[crypto_hash];
	}

	if (crypto_cipher) {
		hdr_size += SALT_SIZE;
		hdr_size += cypher_block_len[crypto_cipher];
	}

	return hdr_size;
}

/*
 * 2.0 packet format:
 *   crypto_cipher_type | crypto_hash_type | __pad0 | __pad1 | hash | salt | data
 *   only data is encrypted, hash only covers salt + data
 *
 * 2.2 packet format
 *   fake_crypto_cipher_type | fake_crypto_hash_type | __pad0 | __pad1 | salt | data | hash
 *   only data is encrypted, hash covers the whole packet
 *
 *  we need to leave fake_* unencrypted for older versions of corosync to reject the packets,
 *  we need to leave __pad0|1 unencrypted for performance reasons (saves at least 2 memcpy and
 *  and extra buffer but values are hashed and verified.
 */

int crypto_encrypt_and_sign (
	struct crypto_instance *instance,
	const unsigned char *buf_in,
	const size_t buf_in_len,
	unsigned char *buf_out,
	size_t *buf_out_len)
{
	struct crypto_config_header *cch = (struct crypto_config_header *)buf_out;
	int err;

	switch (instance->crypto_compat_type) {
		case CRYPTO_COMPAT_2_0:
			cch->crypto_cipher_type = instance->crypto_cipher_type;
			cch->crypto_hash_type = instance->crypto_hash_type;
			cch->__pad0 = 0;
			cch->__pad1 = 0;

			buf_out += sizeof(struct crypto_config_header);

			err = encrypt_and_sign_nss_2_0(instance,
						       buf_in, buf_in_len,
						       buf_out, buf_out_len);

			*buf_out_len = *buf_out_len + sizeof(struct crypto_config_header);
			break;
		case CRYPTO_COMPAT_2_2:
			cch->crypto_cipher_type = CRYPTO_CIPHER_TYPE_2_2;
			cch->crypto_hash_type = CRYPTO_HASH_TYPE_2_2;
			cch->__pad0 = 0;
			cch->__pad1 = 0;

			err = encrypt_and_sign_nss_2_2(instance,
						   buf_in, buf_in_len,
						   buf_out, buf_out_len);
			break;
		default:
			err = -1;
			break;
	}

	return err;
}

int crypto_authenticate_and_decrypt (struct crypto_instance *instance,
	unsigned char *buf,
	int *buf_len)
{
	struct crypto_config_header *cch = (struct crypto_config_header *)buf;

	switch (instance->crypto_compat_type) {
		case CRYPTO_COMPAT_2_0:

			/*
			 * decode crypto config of incoming packets
			 */

			if (cch->crypto_cipher_type != instance->crypto_cipher_type) {
				log_printf(instance->log_level_security,
					   "Incoming packet has different crypto type. Rejecting");
				return -1;
			}

			if (cch->crypto_hash_type != instance->crypto_hash_type) {
				log_printf(instance->log_level_security,
					   "Incoming packet has different hash type. Rejecting");
				return -1;
			}

			if ((cch->__pad0 != 0) || (cch->__pad1 != 0)) {
				log_printf(instance->log_level_security,
					   "Incoming packet appears to have features not supported by this version of corosync. Rejecting");
				return -1;
			}

			/*
			 * invalidate config header and kill it
			 */

			cch = NULL;
			*buf_len -= sizeof(struct crypto_config_header);
			memmove(buf, buf + sizeof(struct crypto_config_header), *buf_len);

			return authenticate_and_decrypt_nss_2_0(instance, buf, buf_len);
			break;
		case CRYPTO_COMPAT_2_2:
			if (cch->crypto_cipher_type != CRYPTO_CIPHER_TYPE_2_2) {
				log_printf(instance->log_level_security,
					   "Incoming packet has different crypto type. Rejecting");
				return -1;
			}

			if (cch->crypto_hash_type != CRYPTO_HASH_TYPE_2_2) {
				log_printf(instance->log_level_security,
					   "Incoming packet has different hash type. Rejecting");
				return -1;
			}

			/*
			 * authenticate packet first
			 */

			if (authenticate_nss_2_2(instance, buf, buf_len) != 0) {
				return -1;
			}

			/*
			 * now we can "trust" the padding bytes/future features
			 */

			if ((cch->__pad0 != 0) || (cch->__pad1 != 0)) {
				log_printf(instance->log_level_security,
					   "Incoming packet appears to have features not supported by this version of corosync. Rejecting");
				return -1;
			}

			/*
			 * decrypt
			 */

			if (decrypt_nss_2_2(instance, buf, buf_len) != 0) {
				return -1;
			}

			/*
			 * invalidate config header and kill it
			 */
			cch = NULL;
			memmove(buf, buf + sizeof(struct crypto_config_header), *buf_len);

			return 0;
			break;
		default:
			return -1;
			break;
	}
}

struct crypto_instance *crypto_init(
	const unsigned char *private_key,
	unsigned int private_key_len,
	const char *crypto_cipher_type,
	const char *crypto_hash_type,
	const char *crypto_compat_type,
	void (*log_printf_func) (
		int level,
		int subsys,
                const char *function,
                const char *file,
                int line,
                const char *format,
                ...)__attribute__((format(printf, 6, 7))),
	int log_level_security,
	int log_level_notice,
	int log_level_error,
	int log_subsys_id)
{
	struct crypto_instance *instance;
	instance = malloc(sizeof(*instance));
	if (instance == NULL) {
		return (NULL);
	}
	memset(instance, 0, sizeof(struct crypto_instance));

	memcpy(instance->private_key, private_key, private_key_len);
	instance->private_key_len = private_key_len;

	instance->crypto_cipher_type = string_to_crypto_cipher_type(crypto_cipher_type);
	instance->crypto_hash_type = string_to_crypto_hash_type(crypto_hash_type);
	instance->crypto_compat_type = string_to_crypto_compat_type(crypto_compat_type);

	instance->crypto_header_size = crypto_sec_header_size(crypto_cipher_type, crypto_hash_type);

	instance->log_printf_func = log_printf_func;
	instance->log_level_security = log_level_security;
	instance->log_level_notice = log_level_notice;
	instance->log_level_error = log_level_error;
	instance->log_subsys_id = log_subsys_id;

	if (init_nss(instance, crypto_cipher_type, crypto_hash_type, crypto_compat_type) < 0) {
		free(instance);
		return(NULL);
	}

	return (instance);
}
