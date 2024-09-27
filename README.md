# Google-Books-Recommendation-Systems-Book-Genie
Prepare to embark on countless literary journeys with Book Genie, your faithful guide to a world of captivating reads!

Welcome to Book Genie, your personal literary wish-granter!

Here's how Book Genie unlocks the perfect books for you:

1. Gathering the Treasures:
Book Genie taps into the vast world of Google Books, gathering a trove of literary gems.
This enchanting information is carefully stored in a special book vault, ready for your exploration.

2. Understanding Your Wishes:
Book Genie meticulously prepares the books for your eyes, transforming text into magical numerical signals.
These signals capture the essence of each book's authors, categories, and descriptions.

3. Finding Your Perfect Match:
When you whisper a wish, Book Genie compares your desires to these signals, using a secret technique called Cosine Similarity.
This unveils the books that align most closely with your unique tastes, like finding the missing puzzle pieces!

4. Preserving the Magic:
To ensure your wishes can always be granted, Book Genie preserves its knowledge in a mystical artifact known as the 'book_recommendation_model.pkl'.

5. Fulfilling Your Literary Dreams:
Whenever you seek a new adventure, Book Genie conjures forth a list of the top 10 books that resonate with your heart.
Simply utter a search query, and watch as the perfect books materialize!


----------------------------------------------------------------------------------------------------------------------

Retrieving Book Data from Google Books API
The initial segment of the code retrieves book data from the Google Books API and saves it in a CSV file named '008106.csv.' The fetch_books function, which takes a search query, API key, and the desired number of results as parameters, executes multiple requests to the Google Books API. It gathers book information and writes it to the CSV file.

Text Processing and Feature Engineering
Subsequently, the code engages in preprocessing the textual data obtained from the fetched books. The preprocess_text function tokenizes, converts to lowercase, removes stop words, and applies stemming to the text. The processed text is then utilized to generate new features, such as Processed_Authors, Processed_Categories, and Combined_Features (a fusion of book description, authors, and categories).

TF-IDF Vectorization
The code employs the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert textual data into numerical vectors. This is accomplished using the TfidfVectorizer from scikit-learn.

Cosine Similarity Calculation
Cosine similarity between books is computed based on their TF-IDF vectors, and the results are stored in the cosine_sim matrix.

Recommendation Function
The recommend_books function processes a search query, preprocesses it, vectorizes it using TF-IDF, and calculates cosine similarity using the precomputed matrix. It then returns the titles of the top 10 recommended books based on similarity.

Model Serialization (Saving and Loading)
The entire model, encompassing the DataFrame, TF-IDF vectorizer, TF-IDF matrix, and cosine similarity matrix, is serialized using the pickle module. The model is saved to a file named 'book_recommendation_model.pkl.' Later, the model can be loaded using pickle to offer book recommendations.

Example Query
The final part of the code illustrates how to utilize the loaded model for recommendations. In this instance, a search query ("java") is employed, and the top recommended books are printed.

This code establishes a book recommendation system using Google Books data, TF-IDF vectorization, and cosine similarity. The model is saved for future use and can be loaded to provide book recommendations based on user queries.
