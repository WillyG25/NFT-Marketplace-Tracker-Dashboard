from hashlib import sha256
import streamlit as st

class UserManager:
    def __init__(self, db_manager, session_state):
        self.db = db_manager
        self.session_state = session_state
        
    def hash_password(self, password):
        hashed_password = sha256(password.encode()).hexdigest()
        return hashed_password
    
    def signup(self, username, email, password, nft_data, collection_data):
        existing_user = self.db.fetch_one(f"SELECT * FROM Users WHERE Username = '{username}'")

        if existing_user:
            st.error('Username already taken. Please choose a different one.')
            return

        existing_email = self.db.fetch_one(f"SELECT * FROM Users WHERE Email = '{email}'")

        if existing_email:
            st.error('Email already in use. Please choose a different one.')
            return

        hashed_password = self.hash_password(password)

        # Step 1: Insert or update Collection data
        self.db.execute_query(
            f"INSERT INTO Collections (Contract_Address, Name, Total_Supply) "
            f"VALUES ('{collection_data['Contract_Address']}', '{collection_data['Name']}', {collection_data['Total_Supply']})"
        )

        # Get the generated Collection_ID
        collection_id = self.db.fetch_one(f"SELECT LAST_INSERT_ID()")[0]

        # Step 2: Insert or update NFT data
        self.db.execute_query(
            f"INSERT INTO NFTs (Contract_Address, Token_Name, Total_Current_Owners, Address, Quantity, Token_ID, Image, chain_name) "
            f"VALUES ('{nft_data['Contract_Address']}', '{nft_data['Token_Name']}', {nft_data['Total_Current_Owners']}, "
            f"'{nft_data['Address']}', {nft_data['Quantity']}, {nft_data['Token_ID']}, '{nft_data['Image']}', '{nft_data['chain_name']}')"
        )

        # Get the generated NFT_ID
        nft_id = self.db.fetch_one(f"SELECT LAST_INSERT_ID()")[0]

        # Step 3: Insert or update Wallet data
        # Ensure that you're not inserting redundant data into Wallets
        existing_wallet = self.db.fetch_one(f"SELECT * FROM Wallets WHERE NFT_Address = '{nft_data['Contract_Address']}' AND Wallet_Address = '{nft_data['Address']}'")

        if not existing_wallet:
            self.db.execute_query(
                f"INSERT INTO Wallets (NFT_Address, Wallet_Address) "
                f"VALUES ('{nft_data['Contract_Address']}', '{nft_data['Address']}')"
            )

        # Step 4: Insert user data
        self.db.execute_query(
            f"INSERT INTO Users (Username, Email, Password, NFT_Address, Wallet_Address) "
            f"VALUES ('{username}', '{email}', '{hashed_password}', '{nft_data['Contract_Address']}', '{nft_data['Address']}') "
            f"ON DUPLICATE KEY UPDATE NFT_Address = '{nft_data['Contract_Address']}', Wallet_Address = '{nft_data['Address']}'"
        )

        st.success('User registered successfully. Please log in.')
        # Assuming you have a method to handle session state and page redirection
        # For example, you might use st.experimental_rerun() or update session_state.page
        # st.session_state.page = 'login'
        # st.experimental_rerun()

    def authenticate(self, username, password):
        hashed_password = self.hash_password(password)
        user_data = self.db.fetch_one(f"SELECT * FROM Users WHERE Username = '{username}' AND Password = '{hashed_password}'")

        if user_data:
            self.session_state.page = 'main_app'
        else:
            st.error('Invalid username or password. Please try again.')
