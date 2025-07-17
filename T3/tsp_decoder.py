from brkga_mp_ipr.types import BaseChromosome
from tsp_instance import TSPInstance

class TSPDecoder():
    """
    Traveling Salesman Problem decoder with port capacity constraints and demands.
    It creates a permutation of nodes induced by the chromosome and computes
    the cost of the tour, adding penalties for demand feasibility violations.
    """

    def __init__(self, instance: TSPInstance, calado_portos=None, demanda_portos=None):
        self.instance = instance
        self.calado_portos = calado_portos or []
        self.demanda_portos = demanda_portos or []

    ###########################################################################

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:
        """
        Given a chromosome, builds a tour and applies penalties for:
        1. Demand feasibility (cargo capacity constraints)
        """

        permutation = sorted((key, index) for index, key in enumerate(chromosome))
        tour = [index for _, index in permutation]

        # Calcular custo básico da rota
        cost = self.instance.distance(tour[0], tour[-1])
        for i in range(len(tour) - 1):
            cost += self.instance.distance(tour[i], tour[i + 1])

        # Penalidades
        PENALIDADE_DEMANDA = 50000.0

        # Verificar viabilidade das demandas (se especificadas)
        if self.demanda_portos and len(self.demanda_portos) > 0:
            if not self._verificar_viabilidade_demandas(tour):
                cost += PENALIDADE_DEMANDA

        return cost

    def _verificar_viabilidade_demandas(self, tour):
        """
        Verifica se a rota é viável considerando as demandas e capacidades dos portos.
        Simula o transporte de carga ao longo da rota, igual às restrições do PLI.
        """
        if not self.demanda_portos or not self.calado_portos:
            return True
        
        carga_atual = 0.0
        
        # Começar do depósito (nó 0) com carga total necessária
        if len(self.demanda_portos) > 1:  # se há demandas definidas
            carga_total = sum(self.demanda_portos[1:])  # excluir depósito
        else:
            return True
        
        # Verificar se conseguimos carregar toda a demanda no depósito
        capacidade_deposito = self.calado_portos[0] if len(self.calado_portos) > 0 else float('inf')
        if carga_total > capacidade_deposito:
            return False
        
        carga_atual = carga_total
        
        # Simular entrega ao longo da rota (excluindo o retorno ao depósito)
        for i, porto in enumerate(tour[1:-1], 1):  # pular depósito inicial e final
            if porto == 0:  # se voltou ao depósito no meio da rota
                continue
                
            # Entregar demanda do porto
            if len(self.demanda_portos) > porto:
                demanda_porto = self.demanda_portos[porto]
                carga_atual -= demanda_porto
                
                # Verificar se a carga não ficou negativa
                if carga_atual < 0:
                    return False
                
                # Verificar se a carga atual não excede a capacidade do porto
                if len(self.calado_portos) > porto:
                    capacidade_porto = self.calado_portos[porto]
                    if carga_atual > capacidade_porto:
                        return False
        
        return True

