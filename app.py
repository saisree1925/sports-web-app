import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.title("🏏 Sports Player Selection Tool")

# ------------------------------
# Step 1: Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload your players CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV Loaded Successfully!")
    st.dataframe(df)

    # ------------------------------
    # Step 2: Generate Performance Score (if missing)
    # ------------------------------
    if 'Performance_Score' not in df.columns:
        df['Performance_Score'] = df['Runs'] + df['Batting_Average']*2 + df['Strike_Rate']*0.5 + df['Wickets']*3
        st.info("Performance_Score column created automatically based on stats.")

    # Features & Target
    features = ['Runs', 'Batting_Average', 'Strike_Rate', 'Wickets']
    target = 'Performance_Score'

    X = df[features]
    y = df[target]

    # Scale features and train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # ------------------------------
    # Step 3: Player Comparison
    # ------------------------------
    st.subheader("Compare Two Players")
    player_names = df['Player'].tolist()
    p1 = st.selectbox("Select First Player", player_names)
    p2 = st.selectbox("Select Second Player", player_names)

    if st.button("Compare Players"):
        p1_data = df[df['Player'] == p1].iloc[0]
        p2_data = df[df['Player'] == p2].iloc[0]
        comparison = pd.DataFrame({
            'Metric': features,
            p1: [p1_data[f] for f in features],
            p2: [p2_data[f] for f in features]
        })
        st.dataframe(comparison)

        # Plot comparison
        comparison_plot = comparison.set_index('Metric')
        st.bar_chart(comparison_plot)

    # ------------------------------
    # Step 4: Predict Player Performance
    # ------------------------------
    st.subheader("Predict Player Performance")
    selected_player = st.selectbox("Select a Player to Predict Performance", player_names)
    if st.button("Predict Performance"):
        player_data = df[df['Player'] == selected_player][features]
        player_scaled = scaler.transform(player_data)
        predicted_score = model.predict(player_scaled)[0]
        st.success(f"{selected_player}'s Predicted Performance Score: {predicted_score:.2f}")

    # ------------------------------
    # Step 5: Predict New Player Stats
    # ------------------------------
    st.subheader("Predict New Player Stats")
    new_player_name = st.text_input("Enter New Player Name")
    new_stats = {}
    for f in features:
        new_stats[f] = st.number_input(f"Enter {f}", value=0)

    if st.button("Predict New Player Performance"):
        if new_player_name.strip() != "":
            new_df = pd.DataFrame([new_stats])
            new_scaled = scaler.transform(new_df)
            score = model.predict(new_scaled)[0]
            st.success(f"{new_player_name}'s Predicted Performance Score: {score:.2f}")
        else:
            st.warning("Please enter a name for the new player.")

    # ------------------------------
    # Step 6: Show Top Players Ranking
    # ------------------------------
    st.subheader("Top Players Ranking")
    df['Predicted_Performance'] = model.predict(X_scaled)
    df_sorted = df.sort_values(by='Predicted_Performance', ascending=False)
    st.dataframe(df_sorted[['Player', 'Predicted_Performance']])

    # Plot top 10 players
    st.bar_chart(df_sorted.head(10).set_index('Player')['Predicted_Performance'])

else:
    st.info("Please upload a CSV file to start.")
