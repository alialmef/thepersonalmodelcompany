"""Thin HTTP routes for action proposal review."""

from __future__ import annotations

from typing import Any

from pmc.actions.adapters.base import ActionExecutionMode
from pmc.actions.service import ActionService, ActionServiceError


def build_actions_router(action_service: ActionService) -> Any:
    from fastapi import APIRouter, HTTPException, Query

    router = APIRouter(prefix="/v1/users/{user_id}", tags=["actions"])

    def translate_error(exc: ActionServiceError) -> HTTPException:
        return HTTPException(status_code=exc.status_code, detail=str(exc))

    @router.get("/actions/capabilities")
    def list_action_capabilities(user_id: str) -> dict[str, Any]:
        try:
            return action_service.list_capabilities(user_id).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals")
    def create_action_proposal(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return action_service.create_proposal(user_id, payload).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.get("/actions/proposals")
    def list_action_proposals(
        user_id: str,
        status: str | None = Query(default=None),
        limit: int = Query(default=20),
    ) -> dict[str, Any]:
        try:
            return action_service.list_proposals(
                user_id,
                status=status,
                limit=limit,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals/{proposal_id}/review")
    def review_action_proposal(
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return action_service.review_proposal(
                user_id,
                proposal_id,
                payload,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals/{proposal_id}/simulate")
    def simulate_action_proposal(
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return action_service.run_proposal(
                user_id,
                proposal_id,
                ActionExecutionMode.SIMULATE,
                payload,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals/{proposal_id}/stage")
    def stage_action_proposal(
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return action_service.run_proposal(
                user_id,
                proposal_id,
                ActionExecutionMode.STAGE,
                payload,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals/{proposal_id}/execute")
    def execute_action_proposal(
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return action_service.run_proposal(
                user_id,
                proposal_id,
                ActionExecutionMode.EXECUTE,
                payload,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/actions/proposals/{proposal_id}/undo")
    def undo_action_proposal(
        user_id: str,
        proposal_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return action_service.run_proposal(
                user_id,
                proposal_id,
                ActionExecutionMode.UNDO,
                payload,
            ).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    @router.post("/verification/action-traces")
    def record_action_trace(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return action_service.record_trace(user_id, payload).model_dump()
        except ActionServiceError as e:
            raise translate_error(e) from e

    return router
