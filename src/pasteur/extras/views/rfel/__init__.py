from functools import partial

import pandas as pd

from ....utils import LazyChunk, LazyFrame, get_relative_fn, to_chunked
from ....view import TabularView, View, filter_by_keys, filter_by_keys_merged


class RfelView(View):
    """The mimic core tables, slightly post processed."""

    def __init__(
        self,
        short_name: str,
        deps: dict[str, list[str]],
        dataset: str | None = None,
        **kwargs,
    ) -> None:
        self.name = f"rfel_{short_name}"
        self.dataset = dataset or self.name
        self.deps = deps
        # Current datasets do not need transformer inter-table references
        self.trn_deps = {}
        self.parameters = get_relative_fn(f"parameters_{short_name}.yml")

        super().__init__(**kwargs)

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        assert len(tables) == 1
        return next(iter(tables.values()))()


class ConsumerExpendituresView(RfelView):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            short_name="ce",
            deps={
                "households": ["households"],
                "expenditures": ["expenditures"],
                "members": ["household_members"],
            },
            **kwargs,
        )

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "households":
                new_df = tables["households"]()
                new_df.index = new_df.index.astype(pd.Int64Dtype())
                new_df.index.name = "household_id"

                # sort by index
                new_df = new_df.sort_index()
                return new_df
            case "expenditures":
                new_df = tables["expenditures"]()
                new_df["household_id"] = new_df["household_id"].astype(pd.Int64Dtype())
                new_df = new_df.drop(columns=["year"])
                new_df.index.name = "expenditure_id"

                # sort by index and then by month
                new_df = new_df.sort_values(by=["household_id", "month"])

                return new_df
            case "members":
                new_df = tables["household_members"]()
                new_df["household_id"] = new_df["household_id"].astype(pd.Int64Dtype())
                new_df = new_df.drop(columns=["year"])
                new_df.index.name = "member_id"
                return new_df
            case other:
                assert False, f"Table {other} not part of view {self.name}"


class StudentLoanView(RfelView):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            short_name="sl",
            deps={
                "person": [
                    "person",
                    "male",
                    "unemployed",
                    "no_payment_due",
                    "filed_for_bankrupcy",
                    "longest_absense_from_school",
                ],
                "enrolled": ["enrolled"],
                "enlist": ["enlist"],
            },
            **kwargs,
        )

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "person":
                df = tables["person"]()
                
                df['male'] = df.index.isin(tables['male']().index)
                df['unemployed'] = df.index.isin(tables['unemployed']().index)
                df['no_payment_due'] = tables['no_payment_due']()['bool'] == 'pos'
                df['filed_for_bankrupcy'] = df.index.isin(tables['filed_for_bankrupcy']().index)
                df['longest_absense_from_school'] = tables['longest_absense_from_school']()

                df.index = df.index.str.replace("student", "").astype(pd.Int64Dtype())

                return df
            case "enrolled":
                df = tables["enrolled"]()
                df['name'] = df['name'].str.replace("student", "").astype(pd.Int64Dtype())
                df.index.name = "enrollment_id"
                return df
            case "enlist":
                df = tables["enlist"]()
                df['name'] = df['name'].str.replace("student", "").astype(pd.Int64Dtype())
                df.index.name = "enlistment_id"
                return df
            case other:
                assert False, f"Table {other} not part of view {self.name}"


class FinancialClientView(RfelView):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            short_name="fnc",
            dataset="rfel_fn",
            deps={
                "client": ["client"],
                "client_district": ["client", "district"],
                "account": ["disp", "account"],
                "card": ["card", "disp"],
                "loan": ["loan", "disp"],
                "order": ["order", "disp"],
                "trans": ["trans", "disp"],
            },
            **kwargs,
        )

    @staticmethod
    def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime.date columns to pd.Timestamp (datetime64)."""
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                val = df[col].dropna().iloc[0] if df[col].notna().any() else None
                if hasattr(val, "year") and not hasattr(val, "hour"):
                    df[col] = pd.to_datetime(df[col])
        return df

    @to_chunked
    def ingest(self, name, **tables: LazyChunk):
        match name:
            case "client":
                df = tables["client"]()
                df.index = df.index.astype(pd.Int64Dtype())
                df.index.name = "client_id"
                df["district_id"] = df["district_id"].astype(pd.Int64Dtype())
                return self._convert_dates(df.sort_index())

            case "client_district":
                client = tables["client"]()
                district = tables["district"]()

                # Look up district for each client via client.district_id
                district.index = district.index.astype(pd.Int64Dtype())
                df = client[["district_id"]].join(
                    district, on="district_id", how="left"
                )
                df = df.drop(columns=["district_id"])
                df.index = df.index.astype(pd.Int64Dtype())
                df.index.name = "client_id"
                return df.sort_index()

            case "account":
                disp = tables["disp"]()
                account = tables["account"]()

                # Join disp with account to get account fields + disp_type
                disp.index = disp.index.astype(pd.Int64Dtype())
                disp["client_id"] = disp["client_id"].astype(pd.Int64Dtype())
                disp["account_id"] = disp["account_id"].astype(pd.Int64Dtype())
                account.index = account.index.astype(pd.Int64Dtype())

                df = disp[["client_id", "account_id", "type"]].rename(
                    columns={"type": "disp_type"}
                )
                df = df.join(account, on="account_id", how="left")
                df["district_id"] = df["district_id"].astype(pd.Int64Dtype())
                # Use disp_id as primary key
                df.index.name = "disp_id"
                return self._convert_dates(df.sort_values(by=["client_id", "account_id"]))

            case "card":
                card = tables["card"]()
                disp = tables["disp"]()

                card.index = card.index.astype(pd.Int64Dtype())
                card["disp_id"] = card["disp_id"].astype(pd.Int64Dtype())
                disp.index = disp.index.astype(pd.Int64Dtype())
                disp["client_id"] = disp["client_id"].astype(pd.Int64Dtype())

                # Add client_id via disp
                df = card.join(
                    disp[["client_id"]], on="disp_id", how="left"
                )
                df.index.name = "card_id"
                return self._convert_dates(df.sort_values(by=["client_id", "disp_id"]))

            case "loan":
                loan = tables["loan"]()
                disp = tables["disp"]()

                loan.index = loan.index.astype(pd.Int64Dtype())
                loan["account_id"] = loan["account_id"].astype(pd.Int64Dtype())
                disp.index = disp.index.astype(pd.Int64Dtype())
                disp.index.name = "disp_id"
                disp["client_id"] = disp["client_id"].astype(pd.Int64Dtype())
                disp["account_id"] = disp["account_id"].astype(pd.Int64Dtype())

                # Join loan to disp via account_id to get client_id and disp_id
                df = loan.merge(
                    disp[["client_id", "account_id"]].reset_index(),
                    on="account_id",
                    how="left",
                )
                df["disp_id"] = df["disp_id"].astype(pd.Int64Dtype())
                df.index.name = "loan_id"
                return self._convert_dates(df.sort_values(by=["client_id", "disp_id"]))

            case "order":
                order = tables["order"]()
                disp = tables["disp"]()

                order.index = order.index.astype(pd.Int64Dtype())
                order["account_id"] = order["account_id"].astype(pd.Int64Dtype())
                disp.index = disp.index.astype(pd.Int64Dtype())
                disp.index.name = "disp_id"
                disp["client_id"] = disp["client_id"].astype(pd.Int64Dtype())
                disp["account_id"] = disp["account_id"].astype(pd.Int64Dtype())

                df = order.merge(
                    disp[["client_id", "account_id"]].reset_index(),
                    on="account_id",
                    how="left",
                )
                df["disp_id"] = df["disp_id"].astype(pd.Int64Dtype())
                df.index.name = "order_id"
                return df.sort_values(by=["client_id", "disp_id"])

            case "trans":
                trans = tables["trans"]()
                disp = tables["disp"]()

                trans.index = trans.index.astype(pd.Int64Dtype())
                trans["account_id"] = trans["account_id"].astype(pd.Int64Dtype())
                disp.index = disp.index.astype(pd.Int64Dtype())
                disp.index.name = "disp_id"
                disp["client_id"] = disp["client_id"].astype(pd.Int64Dtype())
                disp["account_id"] = disp["account_id"].astype(pd.Int64Dtype())

                df = trans.merge(
                    disp[["client_id", "account_id"]].reset_index(),
                    on="account_id",
                    how="left",
                )
                df["disp_id"] = df["disp_id"].astype(pd.Int64Dtype())
                df.index.name = "trans_id"
                return self._convert_dates(df.sort_values(by=["client_id", "disp_id"]))

            case other:
                assert False, f"Table {other} not part of view {self.name}"
