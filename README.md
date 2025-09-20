# Mini RAG + Reranker Sprint

This project implements a small Question-Answering service over a tiny set of industrial and machine safety documents. It starts with a basic similarity search (baseline) and then enhances it with a hybrid and learned reranker to improve the relevance of retrieved information. The service provides short, extractive answers grounded in the retrieved text, along with citations.

## Features

- **Data Ingestion:** Processes PDF documents, chunks them into sensible pieces, and stores them in an SQLite database.
- **Embeddings:** Uses a local `all-MiniLM-L6-v2` Sentence Transformer model to create vector embeddings, stored in ChromaDB.
- **Baseline Search:** Cosine similarity search to retrieve top-k relevant document chunks.
- **Hybrid Reranker:** Blends vector similarity scores with keyword (BM25/FTS) scores for improved ranking.
- **Learned Reranker:** A logistic regression model trained on a few simple features (vector score, keyword score, title match, etc.) to reorder candidate chunks, ensuring better evidence rises to the top.
- **Extractive Answers:** Generates short answers by extracting the most relevant sentence from the top-ranked document chunk.
- **Abstention:** The service abstains from answering if the confidence score of the top-ranked chunk falls below a defined threshold.
- **API:** A FastAPI endpoint for asking questions with different search modes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mitanshu-2004/MiniRag-Reranker.git
    cd MiniRag-Reranker
    ```


2.  **Prepare Data and Ingest:**
    Ensure the `industrial-safety-pdfs.zip` and `sources.json` are in the `data/` directory. The `ingest.py` script will automatically unzip the PDFs, chunk them, create embeddings, and train the reranker model.
    ```bash
    python ingest.py
    ```
    This step might take some time as it processes PDFs, converts to vectors and trains the model.

## How to Run the API

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## API Endpoint

-   **POST `/ask`**
    -   **Request Body:**
        ```json
        {
            "query": "string",
            "top_k": "integer" (default: 5),
            "mode": "string" (options: "baseline", "hybrid", "learned", default: "learned")
        }
        ```
    -   **Response:**
        ```json
        {
            "answer": "string | null",
            "reranker_used": "string",
            "contexts": [
                {
                    "doc_name": "string",
                    "doc_title": "string",
                    "doc_url": "string",
                    "page_num": "integer",
                    "chunk_index": "integer",
                    "score": "float",
                    "content": "string"
                }
            ]
        }
        ```

## Example cURL Requests

### Easy Question (using learned reranker)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the purpose of OSHA 3170?",
           "top_k": 3,
           "mode": "learned"
         }'
```

### Tricky Question (comparing baseline vs. hybrid reranker vs. learned reranker)

```bash
# Baseline search
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Explain the differences between PLd and PLe in ISO 13849-1.",
           "top_k": 5,
           "mode": "baseline"
         }'

# Hybrid reranker
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Explain the differences between PLd and PLe in ISO 13849-1.",
           "top_k": 5,
           "mode": "hybrid"
         }'

# Learned reranker
curl -X POST "http://127.00.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Explain the differences between PLd and PLe in ISO 13849-1.",
           "top_k": 5,
           "mode": "learned"
         }'
```

## Reranker Comparison Script

The `compare_rerankers.py` script allows you to run the 8 test questions (defined in `train_model/questions.py`) across all three reranking modes ("baseline", "hybrid", and "learned") and generates a comparison table. It also exports these results to a CSV file named `reranker_comparison.csv`.

To run the comparison:
```bash
python compare_rerankers.py
```

## Comparison Results Table

| Question | Baseline Answer | Baseline Top Doc | Hybrid Answer | Hybrid Top Doc | Learned Answer | Learned Top Doc |
|---|---|---|---|---|---|---|
| Q1: According to Regulation (EU) 2023/1230, what categories of machinery are considered high-risk and require third-party conformity assessment? | None | safebk-rm002_-en-p.pdf (Score: 0.41) | 3MACHINERY SAFE BOOK 5 Regulations The Machinery Directive The Machinery Directive covers the supply of new machinery and other equipment including safety components. It is an offense to supply machinery within the EU unless the provisions and requirements of the Directive are met. | safebk-rm002_-en-p.pdf (Score: 0.60) | 3MACHINERY SAFE BOOK 5 Regulations The Machinery Directive The Machinery Directive covers the supply of new machinery and other equipment including safety components. It is an offense to supply machinery within the EU unless the provisions and requirements of the Directive are met. | safebk-rm002_-en-p.pdf (Score: 0.69) |
| Q2: What are the key changes introduced in the 2023 edition of EN ISO 13849-1 compared to previous versions? | None | rep0217e.pdf (Score: 0.40) | However, the new directive resulted in control systems becoming more strongly the focus of the safety analysis [7; 8]. In its third, 2015 edition, EN ISO 13849-1 is the successor standard to EN 954-1:1996 [9], and is already listed in the Official Journal of the EU. The presumption of conformity to which the 2008 version gave rise expired on 30 June 2016. | rep0217e.pdf (Score: 0.60) | However, the new directive resulted in control systems becoming more strongly the focus of the safety analysis [7; 8]. In its third, 2015 edition, EN ISO 13849-1 is the successor standard to EN 954-1:1996 [9], and is already listed in the Official Journal of the EU. The presumption of conformity to which the 2008 version gave rise expired on 30 June 2016. | rep0217e.pdf (Score: 0.69) |
| Q3: How does Eaton compare the use of EN ISO 13849-1 and IEC 62061 for functional safety assessments? | None | safebk-rm002_-en-p.pdf (Score: 0.49) | (EN) ISO 13849-1 can be applied to pneumatic, hydraulic, mechanical as well as electrical systems. Joint Technical Report on IEC/EN 62061 and (EN) ISO 13849-1A joint report has been prepared within IEC and ISO to help users of both standards. | safebk-rm002_-en-p.pdf (Score: 0.60) | The essential approach of the standards governing func - tional safety (IEC 61508 and IEC 62061) developed by the International Electrotechnical Commission (IEC), namely that of defining probabilities of failure as the characteris - tic parameter without the specific inclusion of architec - tures, initially appears more universal. The approach of EN ISO 13849-1, however, offers users the facility for developing and evaluating safety functions, ranging from a sensor to an actuator (e.g. a valve), under the umbrella of one standard, even though the functions may involve different technologies. Part 1 of EN ISO 13849 is accompa - nied by a Part 2 with the title of “Validation”. | rep0217e.pdf (Score: 0.69) |
| Q4: According to OSHA 3170, what types of machine guards are recognized for safeguarding employees from amputations? | Safeguarding Equipment and Protecting Employees from Amputations Occupational Safety and Health Administration U.S. Department of Labor OSHA 3170-02R 2007 | osha3170.pdf (Score: 0.61) | Safeguarding Equipment and Protecting Employees from Amputations Occupational Safety and Health Administration U.S. Department of Labor OSHA 3170-02R 2007 | osha3170.pdf (Score: 0.60) | (The U.S. Bureauof Labor Statistics 2005 annual survey data indicat-ed that there were 8,450 non-fatal amputation cases– involving days away from work – for all privateindustry. Approximately forty-four percent (44%) ofall workplace amputations occurred in the manu-facturing sector and the rest occurred across theconstruction, agriculture, wholesale and retail trade,and service industries.) These injuries result fromthe use and care of machines such as saws, press-es, conveyors, and bending, rolling or shapingmachines as well as from powered and non-pow-ered hand tools, forklifts, doors, trash compactorsand during materials handling activities. Anyone responsible for the operation, servicing, and maintenance (also known as use and care) ofmachines (which, for purposes of this publicationinclude | osha3170.pdf (Score: 0.68) |
| Q5: In IFA Report 2/2017e, what is the maximum achievable Performance Level (PL) without using redundancy? | None | safebk-rm002_-en-p.pdf (Score: 0.14) | that PL requirement increases. Performance Level (PL)The performance level is a discrete level that speci ﬁ es the ability of the safety- related parts of the control system to perform a safety function.In order to assess the PL achieved by an implementation of any of the ﬁ ve designated architectures, the following data is required for the system (or subsystem): • MTTF D (mean time to dangerous failure of each channel) • DC (diagnostic coverage). • Architecture (the category) The following diagram shows a graphical method for determining the PL from the combination of these factors. | safebk-rm002_-en-p.pdf (Score: 0.60) | As already mentioned, the architecture imposes the following limita - tions upon certain Categories. These limitations are inten - ded to prevent the component reliability from being over - stated in comparison with the other influencing variables: • In Category B, a maximum PL of b can be attained. • In Category 1, a maximum PL of c can be attained. | rep0217e.pdf (Score: 0.68) |
| Q6: What best practices does Rockwell recommend for transitioning from the Machinery Directive 2006/42/EC to the new Regulation (EU) 2023/1230? | None | oem-sp123_-en-p.pdf (Score: 0.47) | Further Information The Rockwell Automation website will provide further details on the Machinery Regulation (EU) 2023/1230 and also provide updates as the standards covering the new requirements highlighted in this guide are released. If you’re looking to enhance the performance, safety, and productivity of your business, you might want to learn more about the services, tools, and technology available. | oem-sp123_-en-p.pdf (Score: 0.60) | JULY 2026 First report by the member states that assesses the effectiveness of Articles 6(4) and (5) OCTOBER 2026 Member States must notify the European Commission of their penalty rules and measures.Key dates The new Machinery Regulation (EU) 2023/1230 was published in the official journal on 29 June 2023, and entered into force on 19 July 2023, with a 42-month transition to the application date of 20 January 2027. This date is the same for all EU and EFTA countries; and being a regulation and not a directive means it will be adopted at the same time with no modification. | oem-sp123_-en-p.pdf (Score: 0.70) |
| Q7: According to ISO 10218, what are the main safety requirements for collaborative industrial robots? | RIA’s Foundational Standard ANSI/RIA R15.06- 2012, Safety Requirements for Industrial Robots & Robot Systems •U.S. National Adoption of ISO 10218 - 1,2:2011 –10218 Part 1: Safety Requirements for Industrial ROBOTS –10218 Part 2: Safety Requirements for Industrial ROBOT SYSTEMS and Systems Integration | 12-Franklin--RIA-IEEE_2019-05-19_v2.pdf (Score: 0.61) | RIA’s Foundational Standard ANSI/RIA R15.06- 2012, Safety Requirements for Industrial Robots & Robot Systems •U.S. National Adoption of ISO 10218 - 1,2:2011 –10218 Part 1: Safety Requirements for Industrial ROBOTS –10218 Part 2: Safety Requirements for Industrial ROBOT SYSTEMS and Systems Integration | 12-Franklin--RIA-IEEE_2019-05-19_v2.pdf (Score: 0.60) | 12 As per OSHA, it is essential to implement an efficient safeguarding strategy to protect workplaces involving industrial robots. Following are some of the essential steps industrial businesses must take to ensure a safe workplace environment when integrating robots.Safety Considerations for Industrial Robots Safety standards review At the manufacturer level, it is essential to comply with applicable safety regulations and standards when designing robot applications. These guidelines and specifications help create a safe working environment. | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper.pdf (Score: 0.50) |
| Q8: What hazards and risk factors does Hokuyo highlight in its safety guide for industrial robotics applications? | A Safety Guide to Industrial Robotics Hazards HOKUYO USA 2019 Van Buren Ave., Suite A, Indian Trail, NC 28079CONTACT 704-882-3844 info@hokuyo-usa.com | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper.pdf (Score: 0.59) | A Safety Guide to Industrial Robotics Hazards HOKUYO USA 2019 Van Buren Ave., Suite A, Indian Trail, NC 28079CONTACT 704-882-3844 info@hokuyo-usa.com | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper.pdf (Score: 0.60) | By prioritizing safety and adhering to industry standards, we can ensure that the benefits of industrial robotics are realized while minimizing potential hazards and risks. This whitepaper offers an in-depth exploration of the fundamentals of industrial robotics and aims to provide a comprehensive safety guide for addressing hazards and mitigating risks associated with these mach | Hokuyo-USA_-_A_Safety_Guide_to_Industrial_Robotics_Hazards_-_Whitepaper.pdf (Score: 0.59) |

## What I Learned

This sprint gave me a clear understanding of how to build a better RAG system. I learned that reranking plays a big role in improving search results. A basic vector search is a good start, but even a simple reranker, like logistic regression, makes the answers more relevant by looking at more than just similarity scores. Adding extractive answer generation and a way to skip uncertain answers made the system more reliable, since it only returns answers that are supported by the data. Breaking the system into separate parts for ingestion, search, and training also made it easier to build and improve step by step.