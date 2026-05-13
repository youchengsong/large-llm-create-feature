# SimECNY

A simulator of wallet account, transaction, and closure for E-CNY. 

## Data Simulation

### Wallet Creation Simulator

`python open_wallet_distribution_based.py`

Generated Files `accounts.csv`

### Wallet Transaction Simulator

`python generate_static_edges.py` 

`python python simplified_main.py`

Generated Files `wallet_static_edges_distribution_based.csv` `wallet_temporal_transactions_1105_1450.csv`

### Wallet Closure Simulator

`python generate_wallet_close_data.py` 

Generated Files `new_close_3.csv`

## Data Description

### I. Wallet Opening Module Attributes


The wallet opening data file records all relevant information when a wallet account is opened, including basic wallet information, identity authentication information, bank account information, region information, etc.

No. | Attribute Name                 | Chinese Description              | Data Type   | Description
----|-------------------------------|--------------------------------|-------------|------------------------------------------
1   | wallet_id                     | Wallet ID                       | String      | Unique wallet identifier, numeric string
2   | wallet_type                   | Wallet Type                     | Integer     | 1-Corporate wallet, 2-Personal wallet, 3-Merchant wallet
3   | wallet_level                  | Wallet Level                    | Integer     | 1-Class I wallet, 2-Class II wallet, 3-Class III wallet, 4-Class IV wallet
4   | wallet_open_cert              | Opening ID Card Number          | String      | 18-digit ID card number, can be empty (Class IV wallets do not have this field)
5   | wallet_open_cert_expire       | ID Card Expiry Date             | String      | Format: YYYY-MM-DD, can be empty (Class IV wallets do not have this field)
6   | wallet_open_date              | Wallet Opening Date             | String      | Format: YYYY-MM-DD
7   | wallet_open_timestamp         | Wallet Opening Timestamp        | String      | Format: YYYY-MM-DD HH:MM:SS
8   | wallet_open_tel               | Opening Phone Number             | String      | 11-digit phone number
9   | realname_auth_time            | Real-name Authentication Time    | String      | Format: YYYY-MM-DD HH:MM:SS, can be empty (Class IV wallets do not have this field)
10  | accociated_bank_account       | Associated Bank Account (Full)   | String      | Full bank account number, can be empty (Class III and IV wallets do not have this field)
11  | bank_account_number           | Bank Account Number (First 4)    | String      | First 4 digits of bank account number, can be empty (Class III and IV wallets do not have this field)
12  | init_balance                  | Initial Balance                 | Float       | Initial balance when wallet is opened, 2 decimal places
13  | region_code                   | Region Code                     | String      | Region code used to identify the wallet's region
14  | is_abnormal                   | Is Abnormal Account             | Boolean     | True/False or 1/0, indicates whether it is an abnormal account
15  | open_device                   | Opening Device (MAC Address)     | String      | MAC address format (XX:XX:XX:XX:XX:XX), can be empty (if NaN, then opened at counter)
16  | open_ip                       | Opening IP Address              | String      | IP address, can be empty (if NaN, then opened at counter)

Notes:
- Wallet Level Rules:
  * Class I and II wallets: Require all fields (bank card, ID card, authentication time, etc.)
  * Class III wallets: Do not require bank card binding, but require ID card and authentication time
  * Class IV wallets: Do not require bank card, ID card, expiry date, or authentication time
- Counter Opening: Approximately 10% of wallets are opened at counters, in which case open_device and open_ip are empty
- ID Card and Phone Number Reuse: According to configuration, some ID cards and phone numbers can be shared by multiple wallets


###  II. Transaction Module Attributes


The transaction data file records all transaction records between wallets, including basic transaction information, transaction patterns, risk indicators, device information, etc.

No. | Attribute Name                 | Chinese Description              | Data Type   | Description
----|-------------------------------|--------------------------------|-------------|------------------------------------------
1   | tx_id                         | Transaction ID                  | String      | Unique transaction identifier
2   | timestamp                     | Transaction Timestamp           | String      | Format: YYYY-MM-DD HH:MM:SS
3   | src                           | Source Wallet ID                | String      | Unique identifier of the source wallet
4   | dst                           | Destination Wallet ID           | String      | Unique identifier of the destination wallet
5   | amount                        | Transaction Amount              | Float       | Transaction amount, 2 decimal places
6   | transaction_motif             | Transaction Motif Type          | String      | Transaction type, such as: single_transaction, normal_small_high_freq,regular_large_low_freq, class4_laundering,merchant_laundering, online_laundering, small_amount_testing, etc.
7   | motif_id                      | Motif ID                        | String      | Transactions sharing the same transaction pattern share the same motif_id
8   | transaction_mode              | Transaction Mode                | String      | Transaction mode: single_transaction, forward, fan_out, fan_in, one_to_many, many_to_one, many_to_many
9   | is_risk                       | Is Risk Transaction             | String      | '0' indicates normal transaction, '1' indicates risk transaction
10  | is_src_victim                 | Is Source a Victim              | Integer     | 0 or 1, indicates whether the source is a victim account (for abnormal transaction analysis, mainly based on IP and device changes; if different from opening IP/device, it's fraud; if same, it's induced participation)
11  | src_bank_account_number       | Source Bank Account Number      | String      | First 4 digits of source bank account number
12  | dst_bank_account_number       | Destination Bank Account Number | String      | First 4 digits of destination bank account number
13  | src_wallet_level              | Source Wallet Level             | String      | Source wallet level (1, 2, 3, 4)
14  | dst_wallet_level              | Destination Wallet Level        | String      | Destination wallet level (1, 2, 3, 4)
15  | src_device                    | Source Device (MAC Address)     | String      | Source MAC address, format: XX:XX:XX:XX:XX:XX
16  | src_ip                        | Source IP Address               | String      | Source IP address
17  | dst_device                    | Destination Device (MAC Address) | String      | Destination MAC address, format: XX:XX:XX:XX:XX:XX
18  | dst_ip                        | Destination IP Address          | String      | Destination IP address
19  | interval                      | Time Interval                   | Float       | Time interval (seconds) between adjacent transactions within the same motif_id
20  | hour                          | Transaction Hour                | Integer     | Hour when transaction occurred (0-23)

Notes:
- Transaction Motif Types (transaction_motif):
  * Normal transactions: single_transaction, normal_small_high_freq, regular_large_low_freq
  * Abnormal transactions: class4_laundering, merchant_laundering, online_laundering, small_amount_testing (includes is_src_victim victim money laundering patterns)
- Transaction Modes (transaction_mode):
  * single_transaction: Single transaction
  * forward: Forward transaction (appears in pairs)
  * fan_out: One-to-many transfer out
  * fan_in: Many-to-one transfer in
  * one_to_many: One-to-many (includes both transfer in and out)
  * many_to_one: Many-to-one (includes both transfer in and out)
  * many_to_many: Many-to-many (includes both transfer in and out)
- motif_id: All transactions sharing the same transaction pattern share the same motif_id, used for correlation analysis
- is_src_victim: Used to identify whether the source is a victim account in abnormal transactions (e.g., fraud scenarios)
- Device and IP Information: If a wallet does not have device/IP information at opening, it will be randomly generated during transactions


### III. Wallet Closure Module Attributes


The wallet closure data file records relevant information when a wallet is closed, including closure time, closure channel, risk indicators, etc.

No. | Attribute Name                 | Chinese Description              | Data Type   | Description
----|-------------------------------|--------------------------------|-------------|------------------------------------------
1   | wallet_id                     | Wallet ID                       | String      | Unique wallet identifier, corresponds to wallet_id in the opening module
2   | zs_id                         | Closure ID                      | String      | Unique closure record identifier, format: ZS000001, ZS000002, etc.
3   | zs_channel                    | Closure Channel                 | String      | Closure channel, such as: mobile_app, web, counter, etc.
4   | zs_timestamp                  | Closure Timestamp               | String      | Format: YYYY-MM-DD HH:MM:SS, time of closure operation
5   | wallet_open_tel               | Opening Phone Number             | String      | Phone number used when wallet was opened (for correlation analysis)
6   | is_zs_laundering              | Is Abnormal Closure             | Integer     | 0 indicates normal closure, 1 indicates abnormal closure (money laundering related)

Notes:
- Closure Time (zs_timestamp): Closure time should be later than the wallet's last transaction time
- Closure Channel (zs_channel): Identifies the channel through which the user performed the closure operation
- Abnormal Closure (is_zs_laundering):
  * Priority marking for wallets that meet both conditions: is_abnormal=1 and participated in abnormal transactions
  * Secondary marking for wallets that participated in abnormal transactions
  * Tertiary marking for wallets with is_abnormal=1
  * Finally, select other wallets based on risk scores
- wallet_open_tel: Retains the phone number from opening time, used to analyze the historical account opening situation of phone numbers


### IV. Data Relationships


1. Opening Module and Transaction Module:
   - Linked through wallet_id
   - The src and dst fields in the transaction module correspond to wallet_id in the opening module
   - The src_wallet_level and dst_wallet_level in the transaction module correspond to wallet_level in the opening module

2. Opening Module and Closure Module:
   - Linked through wallet_id
   - The wallet_id in the closure module corresponds to wallet_id in the opening module
   - The wallet_open_tel in the closure module corresponds to wallet_open_tel in the opening module

3. Transaction Module and Closure Module:
   - Indirectly linked through wallet_id
   - Closure time should be later than the wallet's last transaction time in the transaction module
