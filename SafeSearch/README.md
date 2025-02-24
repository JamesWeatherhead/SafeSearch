# Safe Search

Safe Search is an on-premises deterministic search tool designed to protect sensitive information. It enables secure searches against a vector database without risking PHI (Protected Health Information) leakage. The tool leverages vector embeddings to perform a safe vector search, retrieves the top three most relevant matches, and then synthesizes these matches into a single search query for external APIs (such as the Google API or Perplexity API). The synthesized query results are then brought back to LLM 1 along with the physician query that contains PHI, where the external results serve as the {context} for the query.

## How It Works

1. **Vector Search:**  
   Safe Search queries a pre-built vector database using vector embeddings of your search terms. This process retrieves the top three matches while ensuring no PHI is exposed.

2. **Query Synthesis:**  
   The top three matches are passed to a secondary language model (LLM 2), which synthesizes them into a unified search query for external search APIs.

3. **Contextual Querying:**  
   The results from external searches are used as context for the primary language model (LLM 1), which handles PHI-related queries securely.

## Key Features

- **On-Premises Processing:**  
  All sensitive operations are handled on-premises to ensure maximum data security.

- **Deterministic and Secure:**  
  Uses deterministic methods to guarantee that PHI is never inadvertently leaked.

- **Modular and Flexible:**  
  Easily integrates with various external search APIs (e.g., Google API, Perplexity API) to fetch comprehensive results.

- **Robust Workflow:**  
  Separates sensitive PHI queries from public data searches using a multi-model architecture.

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JamesWeatherhead/SafeSearch.git
   cd SafeSearch

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. For any issues or suggestions, feel free to open an issue in this repository.

License

This project is licensed under the MIT License. See the LICENSE file for further details.

Contact

For questions or support, please open an issue on GitHub or contact jacweath@utmb.edu.
