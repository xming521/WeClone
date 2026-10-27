[简体中文](../zh/security.md)｜English

# Security

## What leaves your machine

- Building the life portrait: during extraction, organization and summaries, chat content is sent to the model service you configured (Codex or an API). WeClone tells you and asks you to confirm before it starts. Model answers are cached locally, so content already extracted is not sent again.
- Assistant access: the entries an assistant reads go to the model service behind it. At the **All** level this includes private and sensitive entries.
- Fine-tuned clone: training and inference run on your machine. When building training data with online LLM cleaning (`online_llm_clear`) or image-to-text (`vision_api`) turned on, the matching content is sent to the configured API.

## Storage modes

Set this in `settings.jsonc`:

```json
"security_args": {
    "storage_mode": "encrypted"
}
```

- `encrypted`: the template default. Chat data and everything derived from it (the chat snapshot, portrait, training data and fine-tuned model) are stored encrypted, in files ending in `.enc`. Encrypted mode does not support distributed training yet.
- `plaintext`: stored as plain files. Older configurations without `security_args` use plaintext.

The storage mode is fixed at first initialization. To switch modes, regenerate all data from the original chats in a new output directory. Original chats and any plaintext exports you make are not managed by WeClone, so protect them yourself.

## Password

Set a shared password on the first task or the first web visit. It signs you in to the web UI, and in encrypted mode it also unlocks data for command-line tasks.

```bash
weclone-cli decrypt --input <encrypted-file> --output <new-plaintext-file>  # export plaintext
weclone-cli security-change-password                                       # requires the old password, keeps data
weclone-cli server-reset-password                                          # reset a forgotten password
```

Encrypted data cannot be recovered without the password. After a reset, regenerate data from the original chats in a new output directory.

## Keys and tokens

The `.weclone/` directory holds the data key encrypted with your password (`manifest.json`) and the assistant and bot token records (`assistant.sqlite3`, `bots.sqlite3`). Encrypted data cannot be read without it, so keep it and back it up.

- Both tokens are stored only as hashes and shown once when generated. If you lose one, generate a new one.
- The bot token together with `bots.sqlite3` decrypts all data, so keep it as safe as the password. It survives a password change. After a password reset, generate it again.

## Request logs

Requests WeClone sends to model services are logged in `logs/llm_audit/`. Encrypted mode logs only metadata. Plaintext mode logs full requests and replies.
