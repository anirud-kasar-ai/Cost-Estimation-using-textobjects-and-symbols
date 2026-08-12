"""CRUD for the SQLite pricing catalog (mfg + part_number → unit_cost)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import PricingItem
from db.session import get_db
from schemas.pricing import PricingItemCreate, PricingItemOut, PricingItemUpdate

router = APIRouter(prefix="/api/pricing", tags=["pricing"])


@router.get("", response_model=list[PricingItemOut], summary="List pricing catalog")
def list_pricing(db: Session = Depends(get_db)) -> list[PricingItem]:
    return list(
        db.scalars(
            select(PricingItem).order_by(
                PricingItem.category, PricingItem.mfg, PricingItem.part_number
            )
        ).all()
    )


@router.post(
    "",
    response_model=PricingItemOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add a pricing row",
)
def create_pricing(
    payload: PricingItemCreate, db: Session = Depends(get_db)
) -> PricingItem:
    item = PricingItem(**payload.model_dump())
    db.add(item)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pricing already exists for {payload.mfg} / {payload.part_number}.",
        ) from exc
    db.refresh(item)
    return item


@router.patch(
    "/{item_id}",
    response_model=PricingItemOut,
    summary="Update a pricing row",
)
def update_pricing(
    item_id: str, payload: PricingItemUpdate, db: Session = Depends(get_db)
) -> PricingItem:
    item = db.get(PricingItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Pricing item not found.")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate (mfg, part_number).",
        ) from exc
    db.refresh(item)
    return item


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a pricing row",
)
def delete_pricing(item_id: str, db: Session = Depends(get_db)) -> None:
    item = db.get(PricingItem, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Pricing item not found.")
    db.delete(item)
    db.commit()
