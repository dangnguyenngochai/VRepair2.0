/* radare - LGPL - Copyright 2016 - Oscar Salvador */

#include <r_types.h>
#include <r_util.h>
#include <r_lib.h>
#include <r_bin.h>
#include <r_io.h>
#include "bflt/bflt.h"

static void *load_bytes(RBinFile *arch, const ut8 *buf, ut64 sz, ut64 loaddr, Sdb *sdb) {
	if (!buf || !sz || sz == UT64_MAX) {
		return NULL;
	}
	RBuffer *tbuf = r_buf_new ();
	if (!tbuf) {
		return NULL;
	}
	r_buf_set_bytes (tbuf, buf, sz);
	struct r_bin_bflt_obj *res = r_bin_bflt_new_buf (tbuf);
	r_buf_free (tbuf);
	return res ? res : NULL;
}

static int load(RBinFile *arch) {
	const ut8 *bytes = r_buf_buffer (arch->buf);
	ut64 sz = r_buf_size (arch->buf);
	arch->o->bin_obj = load_bytes (arch, bytes, sz, arch->o->loadaddr, arch->sdb);
	return arch->o->bin_obj ? true : false;
}

static RList *entries(RBinFile *arch) {
	struct r_bin_bflt_obj *obj = (struct r_bin_bflt_obj*)arch->o->bin_obj;
	RList *ret;
	RBinAddr *ptr;

	if (!(ret = r_list_newf (free))) {
		return NULL;
	}
	ptr = r_bflt_get_entry (obj);
	if (!ptr) {
		return NULL;
	}
	r_list_append (ret, ptr);
	return ret;
}

static void __patch_reloc(RBuffer *buf, ut32 addr_to_patch, ut32 data_offset) {
	ut8 val[4] = { 0 };
	r_write_le32 (val, data_offset);
	r_buf_write_at (buf, addr_to_patch, (void *)val, sizeof (val));
}

static int search_old_relocation (struct reloc_struct_t *reloc_table,
				  ut32 addr_to_patch, int n_reloc) {
	int i;
	for (i = 0; i < n_reloc; i++) {
		if (addr_to_patch == reloc_table[i].data_offset) {
			return i;
		}
	}
	return -1;
}

static RList *patch_relocs(RBin *b) {
	struct r_bin_bflt_obj *bin = NULL;
	RList *list = NULL;
	RBinObject *obj;
	int i = 0;
	if (!b || !b->iob.io || !b->iob.io->desc) {
		return NULL;
	}
	if (!b->iob.io->cached) {
		eprintf (
			"Warning: please run r2 with -e io.cache=true to patch "
			"relocations\n");
		return list;
	}
	
	obj = r_bin_cur_object (b);
	if (!obj) {
		return NULL;
	}
	bin = obj->bin_obj;
	list = r_list_newf ((RListFree)free);
	if (!list) {
		return NULL;
	}
	if (bin->got_table) {
		struct reloc_struct_t *got_table = bin->got_table;
		for (i = 0; i < bin->n_got; i++) {
			__patch_reloc (bin->b, got_table[i].addr_to_patch,
				       got_table[i].data_offset);
			RBinReloc *reloc = R_NEW0 (RBinReloc);
			if (reloc) {
				reloc->type = R_BIN_RELOC_32;
				reloc->paddr = got_table[i].addr_to_patch;
				reloc->vaddr = reloc->paddr;
				r_list_append (list, reloc);
			}
		}
		R_FREE (bin->got_table);
	}

	if (bin->reloc_table) {
		struct reloc_struct_t *reloc_table = bin->reloc_table;
		for (i = 0; i < bin->hdr->reloc_count; i++) {
			int found = search_old_relocation (reloc_table, 
						reloc_table[i].addr_to_patch, 
						bin->hdr->reloc_count);
			if (found != -1) {
				__patch_reloc (bin->b, reloc_table[found].addr_to_patch, 
							reloc_table[i].data_offset);
			} else {
				__patch_reloc (bin->b, reloc_table[i].addr_to_patch,
							reloc_table[i].data_offset);
			}
			RBinReloc *reloc = R_NEW0 (RBinReloc);
			if (reloc) {
				reloc->type = R_BIN_RELOC_32;
				reloc->paddr = reloc_table[i].addr_to_patch;
				reloc->vaddr = reloc->paddr;
				r_list_append (list, reloc);
			}
		}
		R_FREE (bin->reloc_table);
	}
	b->iob.write_at (b->iob.io, bin->b->base, bin->b->buf, bin->b->length);
	return list;
}

static int get_ngot_entries(struct r_bin_bflt_obj *obj) {
	ut32 data_size = obj->hdr->data_end - obj->hdr->data_start;
	int i = 0, n_got = 0;
	if (data_size > obj->size) {
		return 0;
	}
	for (i = 0, n_got = 0; i < data_size ; i+= 4, n_got++) {
		ut32 entry, offset = obj->hdr->data_start;
		if (offset + i + sizeof (ut32) > obj->size ||
		    offset + i + sizeof (ut32) < offset) {
			return 0;
		}
		int len = r_buf_read_at (obj->b, offset + i, (ut8 *)&entry,
					 sizeof (ut32));
		if (len != sizeof (ut32)) {
			return 0;
		}
		if (!VALID_GOT_ENTRY (entry)) {
			break;
		}
	}
	return n_got;
}

static RList *relocs(RBinFile *arch) {
	struct r_bin_bflt_obj *obj = (struct r_bin_bflt_obj*)arch->o->bin_obj;
	RList *list = r_list_newf ((RListFree)free);
	int i, len, n_got, amount;
	if (!list || !obj) {
		r_list_free (list);
		return NULL;
	}
	if (obj->hdr->flags & FLAT_FLAG_GOTPIC) {
		n_got = get_ngot_entries (obj);
		if (n_got) {
			amount = n_got *  sizeof (ut32);
			if (amount < n_got || amount > UT32_MAX) {
				goto out_error;
			}
			struct reloc_struct_t *got_table = calloc (
				1, n_got * sizeof (struct reloc_struct_t));
			if (got_table) {
				ut32 offset = 0;
				for (i = 0; i < n_got ; offset += 4, i++) {
					ut32 got_entry;
					if (obj->hdr->data_start + offset + 4 > obj->size ||
					    obj->hdr->data_start + offset + 4 < offset) {
						break;
					}
					len = r_buf_read_at (obj->b, obj->hdr->data_start + offset,
								(ut8 *)&got_entry, sizeof (ut32));
					if (!VALID_GOT_ENTRY (got_entry) || len != sizeof (ut32)) {
						break;
					}
					got_table[i].addr_to_patch = got_entry;
					got_table[i].data_offset = got_entry + BFLT_HDR_SIZE;
				}
				obj->n_got = n_got;
				obj->got_table = got_table;
			}
		}
	}

	if (obj->hdr->reloc_count > 0) {
		int n_reloc = obj->hdr->reloc_count; 

		amount = n_reloc * sizeof (struct reloc_struct_t);
		if (amount < n_reloc || amount > UT32_MAX) {
			goto out_error;	
		}
		struct reloc_struct_t *reloc_table = calloc (1, amount + 1);
		if (!reloc_table) {
			goto out_error;
		}
		amount = n_reloc * sizeof (ut32);
		if (amount < n_reloc || amount > UT32_MAX) {
			free (reloc_table);
			goto out_error;
		}
		ut32 *reloc_pointer_table = calloc (1, amount + 1);
		if (!reloc_pointer_table) {
			free (reloc_table);
			goto out_error;
		}
		if (obj->hdr->reloc_start + amount > obj->size ||
		    obj->hdr->reloc_start + amount < amount) {
			free (reloc_table);
			free (reloc_pointer_table);
			goto out_error;
		}
		len = r_buf_read_at (obj->b, obj->hdr->reloc_start,
				     (ut8 *)reloc_pointer_table, amount);
		if (len != amount) {
			free (reloc_table);
			free (reloc_pointer_table);
			goto out_error;
		}
		for (i = 0; i < obj->hdr->reloc_count; i++) {
			//XXX it doesn't take endian as consideration when swapping
			ut32 reloc_offset =
				r_swap_ut32 (reloc_pointer_table[i]) +
				BFLT_HDR_SIZE;

			if (reloc_offset < obj->hdr->bss_end && reloc_offset < obj->size) {
				ut32 reloc_fixed, reloc_data_offset;
				if (reloc_offset + sizeof (ut32) > obj->size ||
				    reloc_offset + sizeof (ut32) < reloc_offset) {
					free (reloc_table);
					free (reloc_pointer_table);
					goto out_error;
				}
				len = r_buf_read_at (obj->b, reloc_offset,
						     (ut8 *)&reloc_fixed,
						     sizeof (ut32));
				if (len != sizeof (ut32)) {
					eprintf ("problem while reading relocation entries\n");
					free (reloc_table);
					free (reloc_pointer_table);
					goto out_error;
				}
				reloc_data_offset = r_swap_ut32 (reloc_fixed) + BFLT_HDR_SIZE;
				reloc_table[i].addr_to_patch = reloc_offset;
				reloc_table[i].data_offset = reloc_data_offset;
	
				RBinReloc *reloc = R_NEW0 (RBinReloc);
				if (reloc) {
					reloc->type = R_BIN_RELOC_32;
					reloc->paddr = reloc_table[i].addr_to_patch;
					reloc->vaddr = reloc->paddr;
					r_list_append (list, reloc);
				}
			}
		}
		free (reloc_pointer_table);
		obj->reloc_table = reloc_table;
	}
	return list;
out_error:
	r_list_free (list);
	return NULL;	
}

static RBinInfo *info(RBinFile *arch) {
	struct r_bin_bflt_obj *obj = NULL; 
	RBinInfo *info = NULL; 
	if (!arch || !arch->o || !arch->o->bin_obj) {
		return NULL;
	}
	obj = (struct r_bin_bflt_obj*)arch->o->bin_obj;
	if (!(info = R_NEW0 (RBinInfo))) {
		return NULL;
	}
	info->file = arch->file ? strdup (arch->file) : NULL;
	info->rclass = strdup ("bflt");
	info->bclass = strdup ("bflt" );
	info->type = strdup ("bFLT (Executable file)");
	info->os = strdup ("Linux");
	info->subsystem = strdup ("Linux");
	info->arch = strdup ("arm");
	info->big_endian = obj->endian;
	info->bits = 32;
	info->has_va = false;
	info->dbg_info = 0;
	info->machine = strdup ("unknown");
	return info;
}

static int check_bytes(const ut8 *buf, ut64 length) {
	return length > 4 && !memcmp (buf, "bFLT", 4);
}

static int check(RBinFile *arch) {
	const ut8 *bytes = arch ? r_buf_buffer (arch->buf) : NULL;
	ut64 sz = arch ? r_buf_size (arch->buf): 0; 
	if (!bytes || !sz) {
		return false;
	}
	return check_bytes (bytes, sz);
}

static int destroy(RBinFile *arch) {
	r_bin_bflt_free ((struct r_bin_bflt_obj*)arch->o->bin_obj);
	return true;
}

RBinPlugin r_bin_plugin_bflt = {
	.name = "bflt",
	.desc = "bFLT format r_bin plugin",
	.license = "LGPL3",
	.load = &load,
	.load_bytes = &load_bytes,
	.destroy = &destroy,
	.check = &check,
	.check_bytes = &check_bytes,
	.entries = &entries,
	.info = &info,
	.relocs = &relocs,
	.patch_relocs = &patch_relocs,
};

#ifndef CORELIB
RLibStruct radare_plugin = {
	.type = R_LIB_TYPE_BIN,
	.data = &r_bin_plugin_bflt,
	.version = R2_VERSION
};
#endif
