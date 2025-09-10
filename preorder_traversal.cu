#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 9
#define E (2 * (N - 1))

// Inclusive scan (Hillis-Steele technique) using shared memory
__global__ void inclusive_scan(int *data, int n) {
    extern __shared__ int s[];
    int t = threadIdx.x;

    // Load into shared memory
    if (t < n) s[t] = data[t];
    __syncthreads();

    // Hillis-Steele inclusive prefix sum
    for (int offset = 1; offset < n; offset <<= 1) {
        int add = (t >= offset) ? s[t - offset] : 0;
        __syncthreads();
        if (t < n) s[t] += add;
        __syncthreads();
    }

    // Write back
    if (t < n) data[t] = s[t];
}

// Build children lists from parent[]
void build_children(const int parent[], int child_count[], int children[][N]) {
    for (int i = 0; i < N; ++i) child_count[i] = 0;
    for (int v = 0; v < N; ++v) {
        int p = parent[v];
        if (p != -1) {
            children[p][ child_count[p]++ ] = v;
        }
    }
}

// Recursive DFS to generate Euler tour edges (down and up)
void dfs_euler(int u, const int parent[], int child_count[], int children[][N],
               int edges_u[], int edges_v[], int *pos) {
    for (int i = 0; i < child_count[u]; ++i) {
        int v = children[u][i];

        // Down edge u -> v
        edges_u[*pos] = u;
        edges_v[*pos] = v;
        (*pos)++;

        dfs_euler(v, parent, child_count, children, edges_u, edges_v, pos);

        // Up edge v -> u
        edges_u[*pos] = v;
        edges_v[*pos] = u;
        (*pos)++;
    }
}

int main() {
    // Labels and parent[] representation
    const char labels[N] = {'a','b','c','d','e','f','g','h','i'};
    int parent[N] = {-1, 0, 0, 1, 1, 2, 3, 3, 4};

    // Build children list from parent[]
    int child_count[N];
    static int children[N][N];
    build_children(parent, child_count, children);

    // Build Euler tour edges using DFS
    int edges_u[E], edges_v[E];
    int pos = 0;
    dfs_euler(0, parent, child_count, children, edges_u, edges_v, &pos);

    if (pos != E) {
        fprintf(stderr, "Warning: Euler tour produced %d edges, expected %d\n", pos, E);
    }

    // Set weights: downward edges = 1, upward edges = 0
    int w[E];
    for (int i = 0; i < E; ++i) {
        w[i] = (parent[edges_v[i]] == edges_u[i]) ? 1 : 0;
    }

    // Device memory allocation
    int *d_w;
    cudaMalloc((void**)&d_w, E * sizeof(int));

    // Copy weights to device
    cudaMemcpy(d_w, w, E * sizeof(int), cudaMemcpyHostToDevice);

    // Launch inclusive scan kernel
    inclusive_scan<<<1, E, E * sizeof(int)>>>(d_w, E);
    cudaDeviceSynchronize();

    // Copy prefix sums back to host
    int pref[E];
    cudaMemcpy(pref, d_w, E * sizeof(int), cudaMemcpyDeviceToHost);

    // Assign preorder numbers during first downward traversal
    int preorder[N];
    int seen[N];
    for (int i = 0; i < N; ++i) {
        preorder[i] = 0;
        seen[i] = 0;
    }

    for (int i = 0; i < E; ++i) {
        if (w[i] == 1) { // downward edge
            int node = edges_v[i];
            if (!seen[node]) {
                preorder[node] = pref[i] + 1; // preorder = prefix + 1
                seen[node] = 1;
            }
        }
    }

    // Ensure root has preorder = 1
    for (int r = 0; r < N; ++r)
        if (parent[r] == -1) {
            preorder[r] = 1;
            break;
        }

    // Print preorder numbering
    printf("\nPreorder numbering:\n");
    for (int i = 0; i < N; ++i) {
        printf("%c -> %d\n", labels[i], preorder[i]);
    }

    // Print preorder traversal order
    printf("\nPreorder Traversal: ");
    for (int k = 1; k <= N; ++k) {
        for (int v = 0; v < N; ++v) {
            if (preorder[v] == k)
                printf("%c ", labels[v]);
        }
    }
    printf("\n");

    // Time and Cost complexity calculation
    int logN = (int)ceil(log2((double)N));
    printf("\nTime Complexity : O(log N) = O(%d)\n", logN);
    printf("Cost Complexity : O(N log N) = O(%d * %d) = O(%d)\n", N, logN, N * logN);

    // Free device memory
    cudaFree(d_w);

    return 0;
}
