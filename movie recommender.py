import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
import re

movies = pd.read_csv('movies.csv') 

movies['title'] = movies['title'].str.strip()

def remove_year_from_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

movies['clean_title'] = movies['title'].apply(remove_year_from_title)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['clean_title']).drop_duplicates()

def get_movie_recommendations(title, num_recommendations=5):
    clean_title = remove_year_from_title(title)
    
    matches = movies['clean_title'].str.contains(clean_title, case=False, na=False)
    
    if not matches.any():
        return []  
    
    idx = indices.iloc[matches.idxmax()]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:num_recommendations + 1]  
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices].tolist()

def on_entry_change(event):
    query = entry.get().strip().lower()
    if not query:
        listbox.place_forget() 
    else:
        cleaned_query = remove_year_from_title(query)
        filtered_titles = [title for title in movies['clean_title'] if cleaned_query in title.lower()]
        listbox.delete(0, tk.END)
        for title in filtered_titles:
            original_title = movies[movies['clean_title'] == title]['title'].values[0]
            listbox.insert(tk.END, original_title)
        listbox.place(x=entry.winfo_x(), y=entry.winfo_y() + entry.winfo_height(), width=entry.winfo_width())  

def on_listbox_select(event):
    selected_title = listbox.get(tk.ANCHOR)
    entry.delete(0, tk.END)
    entry.insert(0, selected_title)
    listbox.place_forget()  

def show_recommendations():
    title = entry.get()
    if title:
        recommendations = get_movie_recommendations(title, 5)
        
        if not recommendations:
            messagebox.showinfo("No Match", f"No match found for '{title}'")
        else:
            recs = "\n".join(recommendations)
            messagebox.showinfo("Recommendations", f"Top 5 similar movies to '{title}':\n\n{recs}")

root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("600x400") 

label = tk.Label(root, text="Enter or select a movie title:", font=("Arial", 12))
label.pack(pady=20)

entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=10)
entry.bind("<KeyRelease>", on_entry_change)  

listbox = tk.Listbox(root, selectmode=tk.SINGLE, height=10, font=("Arial", 12)) 
listbox.bind("<ButtonRelease-1>", on_listbox_select)  

button = tk.Button(root, text="Get Recommendations", command=show_recommendations, font=("Arial", 12))
button.pack(pady=20)
button.pack(side=tk.BOTTOM, pady=20)
root.mainloop()
