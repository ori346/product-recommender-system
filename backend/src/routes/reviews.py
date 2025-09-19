import logging
import os
from typing import List

import requests
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database.models_sql import Product, Review
from models import ProductReview, ProductReviewCreate, ReviewSummarization, ReviewSummary
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["reviews"])

MODEL_ENDPOINT = (
    os.getenv(
        "MODEL_ENDPOINT",
        "http://llama-3-1-8b-instruct-predictor-kickstart-llms.apps.ai-dev02.kni.syseng.devcluster.openshift.com/v1",  # noqa: E501
    )
    + "/chat/completions"
)

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")


async def _check_product_exists(product_id: str, db: AsyncSession) -> None:
    """Check if a product exists, raise 404 if not found"""
    product_stmt = select(Product.item_id).where(Product.item_id == product_id)
    product_result = await db.execute(product_stmt)
    if not product_result.first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID '{product_id}' not found",
        )


async def _fetch_reviews_from_db(
    product_id: str,
    db: AsyncSession,
    limit: int = 1000,
    offset: int = 0,
) -> List[ProductReview]:
    """Internal function to fetch reviews from database"""
    # First check if the product exists
    await _check_product_exists(product_id, db)

    stmt = (
        select(
            Review.id,
            Review.item_id,
            Review.user_id,
            Review.rating,
            Review.title,
            Review.content,
            Review.created_at,
        )
        .where(Review.item_id == product_id)
        .order_by(Review.created_at.desc(), Review.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = (await db.execute(stmt)).all()
    return [
        ProductReview(
            id=row.id,
            productId=row.item_id,
            userId=row.user_id,
            rating=row.rating,
            title=row.title or "",
            comment=row.content or "",
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/{product_id}/reviews", response_model=List[ProductReview])
async def get_reviews_for_product(
    product_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    try:
        return await _fetch_reviews_from_db(product_id, db, limit, offset)
    except Exception as e:
        logger.error(f"Error fetching reviews for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reviews")


@router.get("/{product_id}/reviews/summary", response_model=ReviewSummary)
async def get_reviews_summary(
    product_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        # First check if the product exists
        await _check_product_exists(product_id, db)

        # Get review summary for the product
        stmt = select(func.count(Review.id), func.avg(Review.rating)).where(
            Review.item_id == product_id
        )
        count, avg_rating = (await db.execute(stmt)).one_or_none() or (0, None)
        return ReviewSummary(
            productId=product_id, count=count or 0, avg_rating=float(avg_rating or 0.0)
        )
    except Exception as e:
        logger.error(f"Error computing review summary for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute review summary")


@router.post(
    "/{product_id}/reviews", response_model=ProductReview, status_code=status.HTTP_201_CREATED
)
async def create_review_for_product(
    product_id: str,
    payload: ProductReviewCreate,
    db: AsyncSession = Depends(get_db),
    user=Depends(get_current_user),
):
    try:
        review = Review(
            item_id=product_id,
            user_id=user.user_id,
            rating=payload.rating,
            title=(payload.title or "").strip() or None,
            content=(payload.comment or "").strip() or None,
        )
        db.add(review)
        await db.commit()
        await db.refresh(review)

        return ProductReview(
            id=review.id,
            productId=review.item_id,
            userId=review.user_id,
            rating=review.rating,
            title=review.title or "",
            comment=review.content or "",
            created_at=review.created_at,
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating review for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create review")


@router.get("/{product_id}/reviews/summarize", response_model=ReviewSummarization)
async def summarize_reviews(
    product_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        # Get all reviews for the product
        # TODO: Implement random sampling if too many reviews
        reviews = await _fetch_reviews_from_db(product_id, db, limit=1000)

        if not reviews:
            return ReviewSummarization(
                productId=product_id, summary="No reviews available for this product yet."
            )

        if len(reviews) < 4:
            return ReviewSummarization(
                productId=product_id,
                summary="Not enough reviews available for this product yet.",
            )

        # Prepare review text for summarization
        review_texts = []
        for review in reviews:
            review_text = f"Rating: {review.rating}/5"
            if review.title:
                review_text += f" - Title: {review.title}"
            if review.comment:
                review_text += f" - Comment: {review.comment}"
            review_texts.append(review_text)

        # Combine all reviews into a single text
        combined_reviews = "\n".join(review_texts)

        # Create prompt for Ollama
        prompt = f"""
Please analyze and summarize the following product reviews. Provide a concise summary that highlights:
1. Overall sentiment (positive, negative, mixed)
2. Key strengths mentioned by customers
3. Main concerns or issues raised
4. Overall recommendation

Reviews:
{combined_reviews}

Please provide a clear, concise summary include the adventage disadvantages and 2-3 sentences of conclusion:
"""  # noqa: E501

        response = requests.post(
            MODEL_ENDPOINT,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful, smart buyer who helps customers summarize reviews on electronic shops.",  # noqa: E501
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            headers={"Content-Type": "application/json"},
        )

        # Extract model response from JSON
        model_response = response.json()
        summary = model_response["choices"][0]["message"]["content"].strip()
        return ReviewSummarization(productId=product_id, summary=summary)

    except Exception as e:
        logger.error(f"Error summarizing reviews for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize reviews")
