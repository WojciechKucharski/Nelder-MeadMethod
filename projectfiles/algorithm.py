from my_parser import f
from functools import cache
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List


class NMMethod:
    def __init__(self):
        self.simplexHistory = []

    def runAlgorithm(self, goalFunction: str, P: List[List[float]], alfa: float, beta: float, gamma: float,
                     epsilon: float) -> List[List[float]]:

        n = self.evaluateFunctionDimension(goalFunction)
        d = self.evaluateAverageDistance(P) # useless?

        for i, Pi in enumerate(P):
            if len(Pi) != n:
                raise Exception(f"{i}-th simplex point has wrong dimension")
        if len(P) != (n + 1):
            raise Exception("Not enough starting points")
        iterations = 0
        while True:
            iterations += 1
            self.simplexHistory.append(P)
            if self.stopCriterion(goalFunction, P, epsilon):
                print(iterations)
                break

            F = self.evaluateAllPoints(goalFunction, P)
            h, l = self.get_l_and_h(F)
            P_prim, P_star, P_sstar, P_ssstar = self.getAllP(n, P, h, alfa, beta, gamma)
            Fs = f(goalFunction, P_prim)  # useless?
            Fo = f(goalFunction, P_star)

            if Fo < F[l]:
                Fe = f(goalFunction, P_sstar)
                if Fe < F[l]:
                    P[h] = P_sstar
                else:
                    P[h] = P_star

            elif Fo >= F[l]:

                for i in range(n + 1):
                    if i == h:
                        continue

                    if Fo >= F[i]:
                        if Fo >= F[h]:
                            Fk = f(goalFunction, P_ssstar)
                            if Fk >= F[h]:
                                P = self.simplexReduction(P, n, l)
                                break
                            elif Fk < F[h]:
                                P[h] = P_ssstar
                                break
                        else:
                            P[h] = P_star
                            break

                    elif Fo < F[i]:
                        P[h] = P_star
                        break
        return P

    # STANDARD DEVIATION OF Fi
    def stopCriterion(self, function: str, P: List[List[float]], epsilon: float) -> bool:
        F = self.evaluateAllPoints(function, P)
        if float(np.std(F)) < epsilon:
            return True
        else:
            return False

    def getAllP(self, n: int, P: List[List[float]], h: int, alfa: float, beta: float, gamma: float):
        P_prim = self.centerOfSymmetry(P, h)
        P_star = [(1 - alfa) * P_prim[i] - alfa * P[h][i] for i in range(n)]
        P_sstar = [(1 + gamma) * P_star[i] - gamma * P_prim[i] for i in range(n)]
        P_ssstar = [beta * P[h][i] + (1 - beta) * P_prim[i] for i in range(n)]
        return P_prim, P_star, P_sstar, P_ssstar

    @staticmethod
    def evaluateAllPoints(function: str, P: List[List[float]]) -> List[float]:
        return [f(function, Pi) for Pi in P]

    @staticmethod
    def simplexReduction(P: List[List[float]], n: int, l: int) -> List[List[float]]:
        Pl = P[l].copy()
        for j in range(n + 1):
            P[j] = [(P[j][i] + Pl[i]) / 2 for i in range(n)]
        return P

    @staticmethod
    def get_l_and_h(functionValuesInPoints: List[float]):
        h = functionValuesInPoints.index(max(functionValuesInPoints))
        l = functionValuesInPoints.index(min(functionValuesInPoints))
        return l, h

    @staticmethod
    def centerOfSymmetry(P: List[List[float]], h: int) -> List[float]:
        numberOfPoints = len(P)
        dimensionOfPoints = len(P[-1])
        if numberOfPoints < h - 1:
            raise Exception("h is outside points")
        centerOfSymmetry = [0] * dimensionOfPoints
        for i in range(numberOfPoints):
            if i == h:
                pass
            else:
                for j in range(dimensionOfPoints):
                    centerOfSymmetry[j] += P[i][j] / (numberOfPoints - 1)
        return centerOfSymmetry

    @staticmethod
    def evaluateFunctionDimension(function: str) -> int:
        functionDimension = None
        for i in range(1, 6):
            if f"x{i}" in function:
                functionDimension = i
        return functionDimension

    @staticmethod
    def evaluateAverageDistance(P: List[List[float]]) -> float:
        distance = 0
        numberOfPoints = len(P)
        dimensionOfPoints = len(P[-1])
        for i in range(numberOfPoints):
            distance += 1 / numberOfPoints * sum(
                [(P[i][j] - P[i][j - 1]) ** 2 for j in range(dimensionOfPoints)]) ** 0.5
        return distance
