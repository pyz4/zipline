#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Position Tracking
=================

    +-----------------+----------------------------------------------------+
    | key             | value                                              |
    +=================+====================================================+
    | asset           | the asset held in this position                    |
    +-----------------+----------------------------------------------------+
    | amount          | whole number of shares in the position             |
    +-----------------+----------------------------------------------------+
    | last_sale_price | price at last sale of the asset on the exchange    |
    +-----------------+----------------------------------------------------+
    | cost_basis      | the volume weighted average price paid per share   |
    +-----------------+----------------------------------------------------+

"""

from __future__ import division
from copy import deepcopy
from enum import Enum
from functools import total_ordering
from math import copysign
from textwrap import dedent
import logbook
import numpy as np
import pandas as pd

from zipline.assets import Future
from zipline.finance.transaction import Transaction
from zipline.data.adjustments import Dividend
import zipline.protocol as zp

log = logbook.Logger("Performance")


class LongShort(Enum):
    long = 1
    short = -1

    def __str__(self):
        return self.name


class PnlRealized(object):
    """
    Subclassing pd.DataFrame throws a KeyError: 0 when
    testing equality. Avoid this error by shielding the data as
    an object but preserving many of the features of pd.DataFrame.
    """

    __slots__ = ("_internal_data",)

    def __init__(self, values=None):
        zeros = pd.DataFrame(
            np.zeros((2, 4)),
            columns=[
                "long_term",
                "short_term",
                "qualified_dividend",
                "ordinary_dividend",
            ],
            index=["long", "short"],
        )

        if isinstance(values, dict):
            self._internal_data = zeros
            self._internal_data.update(pd.DataFrame.from_dict(values))
        elif isinstance(values, tuple):
            self._internal_data = zeros
            self.update(*values)
        elif isinstance(values, pd.DataFrame):
            self._internal_data = values
        elif values is None:
            self._internal_data = zeros
        else:
            raise TypeError(
                "`values` must one of None, dictionary, or pandas.DataFrame, not {}".format(
                    type(values)
                )
            )

    @property
    def as_dataframe(self):
        return self._internal_data

    @property
    def as_dict(self):
        return self._internal_data.to_dict()

    @property
    def _constructor(self):
        raise NotImplementedError()

    def __getstate__(self):
        return self._internal_data.to_dict()

    def __setstate__(self, data):
        """define how object is unpickled"""
        self._internal_data = pd.DataFrame(data)

    def __add__(self, other):
        return type(self)(self._internal_data.add(other._internal_data))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            self.__add__(other)

    def __sub__(self, other):
        return type(self)(self._internal_data.subtract(other._internal_data))

    def __rsub__(self, other):
        if other == 0:
            return self
        else:
            self.__add__(other)

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        a, b = self.as_dataframe.align(other.as_dataframe)
        return a.equals(b)

    def __repr__(self):
        template = dedent("""PnlRealized({})""")
        return template.format(self.as_dict)

    def update(self, position_direction, holding_tenure, value):
        self._internal_data.loc[position_direction, holding_tenure] = value


class Position(object):
    __slots__ = "inner_position", "protocol_position"

    def __init__(
        self, asset, amount=0, cost_basis=0.0, last_sale_price=0.0, last_sale_date=None
    ):
        inner = zp.InnerPosition(
            asset=asset,
            amount=amount,
            cost_basis=cost_basis,
            last_sale_price=last_sale_price,
            last_sale_date=last_sale_date,
            pnl_realized=PnlRealized(),
            lots=set(),
        )
        object.__setattr__(self, "inner_position", inner)
        object.__setattr__(self, "protocol_position", zp.Position(inner))

    def __getattr__(self, attr):
        """
        Attributes are stored and retrieved from the inner_position
        """
        return getattr(self.inner_position, attr)

    def __setattr__(self, attr, value):
        setattr(self.inner_position, attr, value)

    @property
    def has_uncollected_pnl(self):
        return np.count_nonzero(self.pnl_realized.as_dataframe.values, axis=None)

    @property
    def transaction_date(self):
        """
        Transaction Date of a position is deemed the weighted average of
        the transaction_dates of each of its constituent lots
        """
        # default values
        tzinfo = "UTC"

        if self.lots:
            tzinfo = max(self.lots).transaction_date.tzinfo

        timedelta = [
            (lot.transaction_date - pd.Timestamp(0, tz=tzinfo)).total_seconds()
            for lot in self.lots
        ]
        weights = [lot.amount for lot in self.lots]
        average_timedelta = np.average(timedelta, weights=weights)

        return (
            pd.Timestamp(0, tz=tzinfo) + pd.to_timedelta(average_timedelta, unit="s")
        ).normalize()

    def collect_pnl_and_reset(self):
        pnl_realized_copy, self.pnl_realized = self.pnl_realized, PnlRealized()
        return pnl_realized_copy

    def earn_dividend(self, dividend):
        """
        Register the number of shares we held at this dividend's ex date so
        that we can pay out the correct amount on the dividend's pay date.

        Parameters
        ------------ 
        dividends : iterable of (asset, amount, pay_date) namedtuples
        """
        if dividend.asset != self.asset:
            raise TypeError("Dividend.asset must match position.asset")

        total_shares = sum([lot.amount for lot in self.lots])
        if self.amount != total_shares:
            raise ValueError(
                "Shares recorded among lots are inconsistent with total shares registered with the position"
            )

        for lot in self.lots:
            proportion = abs(lot.amount) / abs(total_shares)
            # allocated_amount < 0 to show dividend obligation on short positions
            allocated_amount = (
                abs(total_shares)
                * proportion
                * dividend.amount_per_share
                * copysign(1, lot.amount)
            )

            threshold = pd.Timedelta(60, "D")
            holding_period = dividend.ex_date - lot.transaction_date

            tax_status = (
                "qualified"
                if holding_period > threshold and allocated_amount > 0
                else "ordinary"
            )

            dividend = Dividend(
                asset=dividend.asset,
                amount_per_share=dividend.amount_per_share,
                total_amount=allocated_amount,
                ex_date=dividend.ex_date,
                pay_date=dividend.pay_date,
                ledger_status="earned",
                tax_status=tax_status,
            )
            lot._dividends.add(dividend)

        return list(self.lots)

    def earn_stock_dividend(self, stock_dividend):
        """
        Register the number of shares we held at this dividend's ex date so
        that we can pay out the correct amount on the dividend's pay date.

        For example, closing out a position will not affect the number of shares
        received on the pay_date. Note passing a ratio would not work since
        ratio * 0 = 0.
        """
        return {
            "payment_asset": stock_dividend.payment_asset,
            "share_count": np.trunc(self.amount * float(stock_dividend.ratio)),
            "average_transaction_date": self.transaction_date,
        }

    def process_dividends(self, session):
        raise NotImplementedError()

    def process_stock_dividends(
        self, share_count, cost_basis, spot_price, average_transaction_date
    ):
        if not self.amount:
            txn = Transaction(
                asset=self.asset,
                amount=share_count,
                dt=average_transaction_date,
                price=cost_basis,
                order_id=None,
            )
            self.update(txn)
            return PnlRealized()
        else:
            # handle_split treats ratio as new_amount = old_amount / ratio
            ratio = 1 / (1 + share_count / self.amount)
            # a stock dividend is the same thing has a split
            return self.handle_split(
                self.asset, ratio, cost_basis=0.0, spot_price=spot_price
            )

    def handle_fractional_amounts(self, amount, cost_basis, spot_price):
        # handle fractional shares from each lot by aggregating
        # and treating whole shares as a new lot at aggregate cost-basis
        # while cashing out the remaining fractional share.
        whole = np.trunc(amount)
        fraction = amount - whole

        if whole:
            # tack on the full_share_count to the most recently-traded lot
            # operationally, this should work as updating with a transaction
            # with the transaction_date being an average of all the transaction_dates
            # NOTE: I'm not sure how the IRS treats the holding period of these
            # remainder shares
            total_cost_basis = (
                sum([lot.amount * lot.cost_basis for lot in self.lots])
                + amount * cost_basis
            )
            total_shares = sum([lot.amount for lot in self.lots]) + amount
            new_cost_basis = (
                total_cost_basis / total_shares if total_shares > 0 else 0.0
            )

            # create new lot with the position's (average) transaction_date
            txn = Transaction(
                asset=self.asset,
                amount=whole,
                dt=self.transaction_date,
                price=new_cost_basis,
                order_id=None,
                closing_rule=max,
            )

            self.update(txn)

        self.reconcile("cost_basis", "amount")

        cash_realized = PnlRealized()

        # cash out the remaining fractional share
        if fraction:
            cash_amount = round(float(fraction * (spot_price - cost_basis)), 2)
            long_short = LongShort(np.sign(self.amount))
            cash_realized = PnlRealized(
                (str(long_short), "ordinary_dividend", cash_amount)
            )

        return cash_realized

    def handle_split(self, asset, ratio, cost_basis, spot_price):
        """
        Update the position by the split ratio, and return the resulting
        fractional share that will be converted into cash.

        Returns the unused cash.
        """
        if self.asset != asset:
            raise Exception("updating split with the wrong asset!")

        # adjust the # of shares by the ratio
        # (if we had 100 shares, and the ratio is 3,
        #  we now have 33 shares)
        # (old_share_count / ratio = new_share_count)
        # (old_price * ratio = new_price)

        # a split is essentially when new shares are issued at 0 cost basis
        aggregate_leftover_shares = sum(
            [lot.handle_split(ratio, cost_basis=0) for lot in self.lots]
        )
        cash_realized = self.handle_fractional_amounts(
            aggregate_leftover_shares, cost_basis=0.0, spot_price=spot_price
        )
        return cash_realized

    def reconcile(self, *args):

        total_shares = sum([lot.amount for lot in self.lots])
        total_cost_basis = sum([lot.amount * lot.cost_basis for lot in self.lots])

        if "amount" in args:
            setattr(self, "amount", total_shares)

        if "cost_basis" in args:
            try:
                cost_basis = total_cost_basis / total_shares
            except ZeroDivisionError:
                cost_basis = 0.0

            setattr(self, "cost_basis", cost_basis)

    def update(self, txn):
        if self.asset != txn.asset:
            raise Exception("updating position with txn for a " "different asset")

        while True:
            # clean out the store so that we don't fall into infinite recursion
            self._clean_lots()
            lots = self.lots

            # txn is in the same direction as all current lots
            # then, we're not closing out anything and we just open a new lot

            new_lot = Lot(
                txn.asset,
                transaction_date=txn.dt.normalize(),
                amount=txn.amount,
                cost_basis=txn.price,
            )
            txn_direction = copysign(1, txn.amount)

            # default values modified if closing out a lot
            excess = 0
            pnl_realized = PnlRealized()
            # already have the lot (i.e. order on the same day), update that lot
            if new_lot in self.lots:
                # earlier order on same day
                lot = self._get_lot(new_lot)
                lot.update(txn)

            # if no lots to close out, open a new lot
            elif all(copysign(1, x.amount) == txn_direction for x in lots):
                self._add_lot(new_lot)

            # otherwise, try to close out a currently held lot
            else:
                # user selection for lots take precedence
                # but default is FIFO rule
                if txn.target_lots:
                    lot = txn.target_lots.pop(0)
                else:
                    closing_rule = txn.closing_rule or min  # effectively FIFO
                    lot = closing_rule(lots)
                # pass remaining target_lots for later recursive function calls
                _, excess, pnl_realized = lot.update(txn)

            # update position values to maintain compatibility
            total_shares = sum([x.amount for x in lots])
            total_value = sum([x.amount * x.cost_basis for x in lots])

            try:
                self.cost_basis = total_value / total_shares
            except ZeroDivisionError:
                self.cost_basis = 0.0

            # Update the last sale price if txn is
            # best data we have so far
            if self.last_sale_date is None or txn.dt > self.last_sale_date:
                self.last_sale_price = txn.price
                self.last_sale_date = txn.dt

            self.amount = total_shares

            # Update realized pnl for position
            self.pnl_realized = self.pnl_realized + pnl_realized

            # if txn closes outentire lot with excess,
            # then treat excess as a new transaction to be updated with
            if excess:
                new_txn = deepcopy(txn)
                new_txn.amount = excess
                # recursive update
                self.update(new_txn)

            # if no more transactions to update, then end
            break

    def _add_lot(self, lot):
        self.lots.add(lot)

    def _clean_lots(self):
        for lot in self.lots.copy():
            if not abs(lot.amount):
                self.lots.remove(lot)

    def _get_lot(self, lot):
        if lot not in self.lots:
            raise KeyError("{lot} cannot be found in {position}".format(lot, self))

        found = min(filter(lambda x: x == lot, self.lots))
        return found

    def adjust_commission_cost_basis(self, asset, cost):
        """
        A note about cost-basis in zipline: all positions are considered
        to share a cost basis, even if they were executed in different
        transactions with different commission costs, different prices, etc.

        Due to limitations about how zipline handles positions, zipline will
        currently spread an externally-delivered commission charge across
        all shares in a position.
        """

        if asset != self.asset:
            raise Exception("Updating a commission for a different asset?")
        if cost == 0.0:
            return

        # If we no longer hold this position, there is no cost basis to
        # adjust.
        if self.amount == 0:
            return

        # We treat cost basis as the share price where we have broken even.
        # For longs, commissions cause a relatively straight forward increase
        # in the cost basis.
        #
        # For shorts, you actually want to decrease the cost basis because you
        # break even and earn a profit when the share price decreases.
        #
        # Shorts are represented as having a negative `amount`.
        #
        # The multiplication and division by `amount` cancel out leaving the
        # cost_basis positive, while subtracting the commission.

        prev_cost = self.cost_basis * self.amount
        if isinstance(asset, Future):
            cost_to_use = cost / asset.price_multiplier
        else:
            cost_to_use = cost
        new_cost = prev_cost + cost_to_use
        self.cost_basis = new_cost / self.amount

        # NOTE: to ensure the aggregate cost basis per share of the lots is
        # consistent with the over-all position cost basis per share,
        # apply the same adjustment to the latest acquired lot.
        adjustment = cost_to_use / self.amount
        max(self.lots).adjust_cost_basis(adjustment)

    def __repr__(self):
        template = dedent(
            "asset: {asset}, amount: {amount}, \
                cost_basis: {cost_basis}, \
                last_sale_price: {last_sale_price}"
        )
        return template.format(
            asset=self.asset,
            amount=self.amount,
            cost_basis=self.cost_basis,
            last_sale_price=self.last_sale_price,
        )

    def to_dict(self):
        """
        Creates a dictionary representing the state of this position.
        Returns a dict object of the form:
        """
        return {
            "sid": self.asset,
            "amount": self.amount,
            "cost_basis": self.cost_basis,
            "last_sale_price": self.last_sale_price,
            "lots": self.lots,
            "transaction_date": self.transaction_date,
        }


@total_ordering
class Lot(object):

    __slots__ = ("amount", "asset", "cost_basis", "transaction_date", "_dividends")

    def __init__(self, asset, transaction_date, amount, cost_basis):
        self.asset = asset
        self.transaction_date = transaction_date
        self.amount = amount
        self.cost_basis = cost_basis

        self._dividends = set()

    def __eq__(self, other):
        return (self.asset, self.transaction_date) == (
            other.asset,
            other.transaction_date,
        )

    def __lt__(self, other):
        return self.transaction_date < other.transaction_date

    def __hash__(self):
        return hash((self.asset, self.transaction_date))

    def __repr__(self):
        template = dedent(
            "Lot(asset={asset}, \
             transaction_date={transaction_date}, amount={amount}, \
             cost_basis={cost_basis})"
        )
        return template.format(
            asset=self.asset,
            transaction_date=self.transaction_date,
            amount=self.amount,
            cost_basis=self.cost_basis,
        )

    @property
    def earned_dividends(self):
        return filter(lambda div: div.ledger_status == "earned", self._dividends)

    @property
    def paid_dividends(self):
        return filter(lambda div: div.ledger_status == "paid", self._dividends)

    def adjust_cost_basis(self, adjustment):
        self.cost_basis += adjustment

    def handle_split(self, ratio, cost_basis):
        """
        Update the position by the split ratio, and return the resulting
        fractional share that will be converted into cash.

        Returns the unused cash.
        """
        # adjust the # of shares by the ratio
        # (if we had 100 shares, and the ratio is 3,
        #  we now have 33 shares)
        # (old_share_count / ratio = new_share_count)
        # (old_price * ratio = new_price)

        old_cost_basis_total = self.amount * self.cost_basis
        old_shares = self.amount

        # e.g., 33.333
        new_shares_raw = self.amount / float(ratio) - self.amount

        # e.g., 33
        new_shares_whole = np.trunc(new_shares_raw)

        # e.g., 0.333
        fractional_share_count = new_shares_raw - new_shares_whole

        # adjust the cost basis to the nearest cent, e.g., 60.0
        new_cost_basis_total = old_cost_basis_total + new_shares_whole * cost_basis
        new_cost_basis = new_cost_basis_total / (old_shares + new_shares_whole)

        self.cost_basis = new_cost_basis
        self.amount += new_shares_whole

        return fractional_share_count

    def process_dividend_payment(self, pay_date):
        pnl_realized = PnlRealized()
        for div in self.earned_dividends:
            if div.pay_date == pay_date:
                div.update_paid()

                # return a PnlRealized instance to keep ledger consistent.
                # Div.total_amount is negative for short positions, representing
                # short position holder's obligation to pay the dividend
                direction = "long" if div.total_amount > 0 else "short"
                tax_status = div.tax_status
                pnl_realized += PnlRealized(
                    {"{}_dividend".format(tax_status): {direction: div.total_amount}}
                )

        return pnl_realized

    def update(self, txn):
        """
        Lots are mostly immutable: 
        they can only be partially or fully closed out but never expanded.
        """
        if self.asset != txn.asset:
            raise Exception("updating lot with a transaction for a different asset")

        # default return values
        cleared = 0
        excess = 0
        cost_basis_adjustment_total = 0
        pnl_realized = PnlRealized()

        # if in the same direction, add to the current lot
        # otherwise, treat it as a close out, excess amounts are
        # treated as a new transaction
        if copysign(1, txn.amount) == copysign(1, self.amount):
            cleared = txn.amount
            cost_basis_adjustment_total = txn.amount * txn.price
        else:
            # cleared amounts are shares that were transacted
            cleared = copysign(min(self.amount, txn.amount, key=abs), txn.amount)
            excess = (
                self.amount + txn.amount if abs(txn.amount) > abs(self.amount) else 0
            )
            cost_basis_adjustment_total = cleared * self.cost_basis

        # first process cleared amounts and adjust basis
        # if fully closed out, basis will calculate to 0
        total_shares = self.amount + cleared

        # calculate pnl based on what has been cleared before the cost_basis as been adjusted
        # pnl calculations should account for shorts as well
        # pnl is realized only when positions are closed out
        if copysign(1, txn.amount) != copysign(1, self.amount):
            over_year_long = self.transaction_date < txn.dt - pd.Timedelta(days=365)
            holding_tenure = "long_term" if over_year_long else "short_term"
            closed = copysign(cleared, self.amount)  # the shares that were closed out
            closed_direction = "long" if closed > 0 else "short"

            pnl = closed * (txn.price - self.cost_basis)
            pnl_realized.update(
                position_direction=closed_direction,
                holding_tenure=holding_tenure,
                value=pnl,
            )

        # update rest of the metrics
        prev_cost = self.cost_basis * self.amount
        total_cost = prev_cost + cost_basis_adjustment_total
        try:
            self.cost_basis = total_cost / total_shares
        except ZeroDivisionError:
            self.cost_basis = 0.0

        self.amount = self.amount + cleared

        return cleared, excess, pnl_realized
