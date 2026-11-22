"""api/deps.py — FastAPI dependencies that hand routes their interfaces.

Routes declare what they need (a CustomerRepository, a RecommendationRepository),
never a database client. Production gets the Postgres implementations; tests swap
them with `app.dependency_overrides[get_customer_repo] = lambda: FakeRepo()`.
"""

from churn_data.ports import CustomerRepository, RecommendationRepository


def get_customer_repo() -> CustomerRepository:
    from churn_data.db import PgCustomerRepository
    return PgCustomerRepository()


def get_recommendation_repo() -> RecommendationRepository:
    from churn_data.db import PgRecommendationRepository
    return PgRecommendationRepository()
