use std::ffi::{CStr, CString, c_char};

use tokenizers::Tokenizer;

pub struct TokenizerHandle {
    inner: Tokenizer,
}

/// Load a tokenizer from a tokenizer.json path. Returns NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut TokenizerHandle {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path_str = unsafe { CStr::from_ptr(path) }.to_str().unwrap_or("");
    match Tokenizer::from_file(path_str) {
        Ok(tok) => Box::into_raw(Box::new(TokenizerHandle { inner: tok })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a tokenizer returned by tokenizer_from_file.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Encode text to token ids. Writes min(count, capacity) ids into ids_out.
/// Returns actual token count, or -1 on error. Retry with a larger buffer if
/// the return value exceeds capacity.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_encode(
    handle: *const TokenizerHandle,
    text: *const c_char,
    add_special_tokens: bool,
    ids_out: *mut i32,
    capacity: i32,
) -> i32 {
    if handle.is_null() || text.is_null() || ids_out.is_null() {
        return -1;
    }
    let tok = unsafe { &(*handle).inner };
    let text_str = unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("");
    match tok.encode(text_str, add_special_tokens) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            let count = ids.len() as i32;
            let to_copy = ids.len().min(capacity as usize);
            let out = unsafe { std::slice::from_raw_parts_mut(ids_out, to_copy) };
            for (i, &id) in ids[..to_copy].iter().enumerate() {
                out[i] = id as i32;
            }
            count
        }
        Err(_) => -1,
    }
}

/// Decode token ids to text. Returns a malloc'd UTF-8 string; free with
/// tokenizer_free_string(). Returns NULL on error.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_decode(
    handle: *const TokenizerHandle,
    ids: *const i32,
    ids_len: i32,
    skip_special_tokens: bool,
) -> *mut c_char {
    if handle.is_null() || ids.is_null() || ids_len < 0 {
        return std::ptr::null_mut();
    }
    let tok = unsafe { &(*handle).inner };
    let slice = unsafe { std::slice::from_raw_parts(ids, ids_len as usize) };
    let u32_ids: Vec<u32> = slice.iter().map(|&id| id as u32).collect();
    match tok.decode(&u32_ids, skip_special_tokens) {
        Ok(text) => match CString::new(text) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string returned by tokenizer_decode or tokenizer_id_to_token.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}

/// Return the vocabulary size (including added tokens).
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_vocab_size(handle: *const TokenizerHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).inner.get_vocab_size(true) as i32 }
}

/// Look up a token string in the vocabulary. Returns -1 if not found.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_token_to_id(
    handle: *const TokenizerHandle,
    token: *const c_char,
) -> i32 {
    if handle.is_null() || token.is_null() {
        return -1;
    }
    let tok = unsafe { &(*handle).inner };
    let token_str = unsafe { CStr::from_ptr(token) }.to_str().unwrap_or("");
    match tok.token_to_id(token_str) {
        Some(id) => id as i32,
        None => -1,
    }
}

/// Look up a token id in the vocabulary. Returns a malloc'd string; free with
/// tokenizer_free_string(). Returns NULL if the id is not in the vocabulary.
#[unsafe(no_mangle)]
pub extern "C" fn tokenizer_id_to_token(
    handle: *const TokenizerHandle,
    id: i32,
) -> *mut c_char {
    if handle.is_null() || id < 0 {
        return std::ptr::null_mut();
    }
    let tok = unsafe { &(*handle).inner };
    match tok.id_to_token(id as u32) {
        Some(token) => match CString::new(token) {
            Ok(cs) => cs.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}
