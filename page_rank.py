import numpy as np


def get_graph():
    return np.asmatrix([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])


def page_rank_by_power_iteration(decay=0.85):
    graph_aux_matrix = get_graph()

    # Calculate Transition probability matrix
    degree_count = np.matrix.sum(graph_aux_matrix, axis=1)
    transition_matrix = graph_aux_matrix / degree_count

    # Division by zero yields Nan value in place of zero . Below line code solves this problem
    transition_matrix[np.isnan(transition_matrix)] = 0

    # Total number of nodes = row length OR column length, ie graph_aux_matrix.shape[1] = column length
    n = np.sum(graph_aux_matrix.shape[1])

    # initialize with equal probability for a random visitor to visit each node. i.e. 1/N
    page_ranks = np.full((n, 1), 1 / n)

    m_matrix = transition_matrix.transpose()

    # Node 1 is a dangling node.
    # We need to handle that.
    # So we initialized a stochastic distribution of 1 to all dangling nodes as 1/n
    for i in range(m_matrix.shape[1]):
        col = m_matrix[:, i]
        if np.sum(col) == 0:
            m_matrix[:, i] = page_ranks

    # google_matrix = decay * m_matrix + np.ones(n, n) * (1 - decay) / n

    while True:

        new_page_ranks = decay * m_matrix * page_ranks + ((1 - decay) / n) * np.ones((n, 1))

        if np.sum(np.abs(new_page_ranks - page_ranks)) < 10 ** -6:
            break
        page_ranks = new_page_ranks

    print('-----Page Rank by Power Iteration method-----')
    print(page_ranks.round(3))


def personalized_page_rank_by_power_iteration(decay=0.85, query_node=1):
    graph_aux_matrix = get_graph()

    # Calculate Transition probability matrix
    degree_count = np.matrix.sum(graph_aux_matrix, axis=1)
    transition_matrix = graph_aux_matrix / degree_count

    # Division by zero yields Nan value in place of zero . Below line code solves this problem
    transition_matrix[np.isnan(transition_matrix)] = 0

    # Total number of nodes = row length OR column length, ie graph_aux_matrix.shape[1] = column length
    n = np.sum(graph_aux_matrix.shape[1])

    # initialize with equal probability for a random visitor to visit each node. i.e. 1/N
    page_ranks = np.full((n, 1), 1 / n)

    m_matrix = transition_matrix.transpose()

    q_vector = np.zeros((n, 1))
    q_vector[query_node - 1] = 1

    while True:

        new_page_ranks = decay * m_matrix * page_ranks + (1 - decay) * q_vector

        if np.sum(np.abs(new_page_ranks - page_ranks)) < 10 ** -6:
            break
        page_ranks = new_page_ranks

    print('Query Node: ' + str(query_node) + ' PPR:' + str(page_ranks.round(3)[query_node - 1]))


def inverse_matrix_method_pr(decay=0.85):
    graph_aux_matrix = get_graph()

    # Calculate Transition probability matrix
    degree_count = np.matrix.sum(graph_aux_matrix, axis=1)
    transition_matrix = graph_aux_matrix / degree_count

    # Division by zero yields Nan value in place of zero . Below line code solves this problem
    transition_matrix[np.isnan(transition_matrix)] = 0

    # Total number of nodes = row length OR column length, ie graph_aux_matrix.shape[1] = column length
    n = np.sum(graph_aux_matrix.shape[1])

    # initialize with equal probability for a random visitor to visit each node. i.e. 1/N
    page_ranks = np.full((n, 1), 1 / n)

    m_matrix = transition_matrix.transpose()

    # Node 1 is a dangling node.
    # We need to handle that.
    # So we initialized a stochastic distribution of 1 to all nodes as 1/n
    for i in range(m_matrix.shape[1]):
        col = m_matrix[:, i]
        if np.sum(col) == 0:
            m_matrix[:, i] = page_ranks

    page_ranks = np.linalg.inv(np.identity(n) - decay * m_matrix) * ((1 - decay) / n * np.ones((n, 1)))
    print('------Page Rank by Inverse Matrix method------')
    print(page_ranks.round(3))


def inverse_matrix_method_ppr(decay=0.85,query_node=1):
    graph_aux_matrix = get_graph()

    # Calculate Transition probability matrix
    degree_count = np.matrix.sum(graph_aux_matrix, axis=1)
    transition_matrix = graph_aux_matrix / degree_count

    # Division by zero yields Nan value in place of zero . Below line code solves this problem
    transition_matrix[np.isnan(transition_matrix)] = 0

    # Total number of nodes = row length OR column length, ie graph_aux_matrix.shape[1] = column length
    n = np.sum(graph_aux_matrix.shape[1])

    # initialize with equal probability for a random visitor to visit each node. i.e. 1/N
    page_ranks = np.full((n, 1), 1 / n)

    m_matrix = transition_matrix.transpose()

    q_vector = np.zeros((n, 1))
    q_vector[query_node - 1] = 1

    page_ranks = np.linalg.inv(np.identity(n) - decay * m_matrix) * (1 - decay) * q_vector
    print('Query Node: ' + str(query_node) + ' PPR:' + str(page_ranks.round(3)[query_node - 1]))


def main():
    # Just to disable some unnecessary warning
    np.seterr(divide='ignore', invalid='ignore')
    #
    page_rank_by_power_iteration(decay=0.85)
    #
    # inverse_matrix_method_pr(decay=0.85)
    #
    print('-------Personalized Page Rank by Power Iteration method-------')
    # personalized_page_rank_by_power_iteration(query_node=1, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=2, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=3, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=4, decay=0.85)
    personalized_page_rank_by_power_iteration(query_node=5, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=6, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=7, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=8, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=9, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=10, decay=0.85)
    # personalized_page_rank_by_power_iteration(query_node=11, decay=0.85)
    # #
    # print('-------Personalized Page Rank by Matrix Inverse method-------')
    # inverse_matrix_method_ppr(query_node=1, decay=0.85)
    # inverse_matrix_method_ppr(query_node=2, decay=0.85)
    # inverse_matrix_method_ppr(query_node=3, decay=0.85)
    # inverse_matrix_method_ppr(query_node=4, decay=0.85)
    # inverse_matrix_method_ppr(query_node=5, decay=0.85)
    # inverse_matrix_method_ppr(query_node=6, decay=0.85)
    # inverse_matrix_method_ppr(query_node=7, decay=0.85)
    # inverse_matrix_method_ppr(query_node=8, decay=0.85)
    # inverse_matrix_method_ppr(query_node=9, decay=0.85)
    # inverse_matrix_method_ppr(query_node=10, decay=0.85)
    # inverse_matrix_method_ppr(query_node=11, decay=0.85)
    # #/

if __name__ == '__main__':
    main()
