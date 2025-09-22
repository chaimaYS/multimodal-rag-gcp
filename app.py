"""
Multimodal RAG Search — Streamlit UI
Search and filter images from a large database using natural language or image queries.
"""

import streamlit as st
import yaml
import logging
from pathlib import Path
from src.rag_pipeline import MultimodalRAGPipeline
from src.bigquery_vector_store import BigQueryVectorStore

logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("config/config.yml")


@st.cache_resource
def load_pipeline():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return MultimodalRAGPipeline(
        project_id=config["gcp"]["project_id"],
        dataset_id=config["gcp"]["dataset_id"],
        gemini_api_key=config["gemini"]["api_key"],
    )


@st.cache_resource
def load_store():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return BigQueryVectorStore(
        config["gcp"]["project_id"],
        config["gcp"]["dataset_id"],
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### Filters")
        top_k = st.slider("Results to return", 1, 20, 5)
        st.markdown("---")
        st.markdown("### Filter by tag")
        available_tags = [
            "vehicle", "damage", "interior", "exterior", "engine",
            "tire", "windshield", "bumper", "door", "hood",
            "industrial", "warehouse", "equipment", "safety",
        ]
        selected_tags = st.multiselect("Tags", available_tags)
        st.markdown("---")
        st.markdown("### Search mode")
        search_mode = st.radio("Mode", ["Text query", "Image upload"], index=0)
        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "Multimodal RAG system using CLIP embeddings, "
            "BigQuery vector search, and Gemini for answer generation."
        )
    return top_k, selected_tags or None, search_mode


def render_results(result):
    st.markdown(f"### Answer")
    st.info(result.answer)

    if result.retrieved_images:
        st.markdown(f"### Retrieved images ({result.num_results})")
        cols = st.columns(min(3, len(result.retrieved_images)))
        for i, img in enumerate(result.retrieved_images):
            with cols[i % 3]:
                source = img.get("source_path", "")
                score = img.get("similarity_score", 0)
                summary = img.get("summary", "No summary")
                tags = ", ".join(img.get("tags", []))

                if source.startswith("gs://") or source.startswith("http"):
                    st.image(source, use_container_width=True)
                else:
                    st.markdown(f"**{Path(source).name}**")

                st.caption(f"Score: {score:.3f}")
                st.markdown(f"_{summary[:120]}{'...' if len(summary) > 120 else ''}_")
                if tags:
                    st.markdown(f"`{tags}`")
                st.markdown("---")


def render_stats(store):
    stats = store.get_collection_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total images", f"{stats.get('total_images', 0):,}")
        c2.metric("Unique tags", f"{stats.get('unique_tags', 0):,}")
        c3.metric("Avg embedding dim", stats.get("embedding_dim", 512))


def main():
    st.set_page_config(
        page_title="Multimodal RAG Search",
        page_icon="🔍",
        layout="wide",
    )

    st.markdown(
        """
        # 🔍 Multimodal RAG Search
        Search through thousands of images using natural language.
        Powered by CLIP embeddings, BigQuery vector search, and Gemini.
        """
    )

    top_k, tags, search_mode = render_sidebar()

    pipeline = load_pipeline()
    store = load_store()

    render_stats(store)

    st.markdown("---")

    if search_mode == "Text query":
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g. 'damaged front bumper on a silver sedan'",
        )
        if query:
            with st.spinner("Searching..."):
                result = pipeline.query(query, top_k=top_k, filter_tags=tags)
            render_results(result)

    else:
        uploaded = st.file_uploader(
            "Upload an image to find similar ones",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded:
            st.image(uploaded, width=300, caption="Query image")
            with st.spinner("Encoding and searching..."):
                image_embedding = pipeline.encoder.encode_image_from_bytes(
                    uploaded.getvalue()
                )
                retrieved = store.vector_search(
                    query_embedding=image_embedding,
                    top_k=top_k,
                    filter_tags=tags,
                )
            if retrieved:
                st.markdown(f"### Similar images ({len(retrieved)})")
                cols = st.columns(min(3, len(retrieved)))
                for i, img in enumerate(retrieved):
                    with cols[i % 3]:
                        source = img.get("source_path", "")
                        score = img.get("similarity_score", 0)
                        if source.startswith("gs://") or source.startswith("http"):
                            st.image(source, use_container_width=True)
                        st.caption(f"Score: {score:.3f}")
                        st.markdown(f"_{img.get('summary', '')[:100]}_")
            else:
                st.warning("No similar images found.")


if __name__ == "__main__":
    main()
