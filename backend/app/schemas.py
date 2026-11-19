from typing import Optional
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=4)
    password: str = Field(..., min_length=4)


class RegisterAccountRequest(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=80)
    cnic: str = Field(..., min_length=13, max_length=15)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=4)
    city: str = Field(..., min_length=2)
    monthly_income: float = Field(..., gt=0)
    employment_status: str = Field(default="Salaried")
    account_type: str = Field(default="Current")
    initial_deposit: float = Field(default=50000.0, ge=1000)


class TransferRequest(BaseModel):
    sender_account_id: int
    beneficiary_bank: str = Field(..., min_length=2, max_length=60)
    beneficiary_account_number: str = Field(..., min_length=4, max_length=40)
    beneficiary_title: Optional[str] = ""
    amount: float = Field(..., gt=0, description="Transfer amount in PKR")
    txn_city: str = Field(..., min_length=2, max_length=50)
    txn_hour: int = Field(..., ge=0, le=23)


class DepositRequest(BaseModel):
    account_id: int
    amount: float = Field(..., gt=0)
    deposit_source: str = Field(default="Cash Deposit / ATM CDM")


class LoanEvaluationRequest(BaseModel):
    customer_id: int
    requested_amount: float = Field(..., gt=10000)
    tenure_months: int = Field(..., ge=6, le=120)
    existing_debt: float = Field(default=0.0, ge=0)
    purpose: str = Field(..., min_length=3, max_length=100)


class UnfreezeRequest(BaseModel):
    account_id: int
    admin_note: Optional[str] = "Verified identity with customer via OTP"