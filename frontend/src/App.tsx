import { useInfiniteQuery } from '@tanstack/react-query'

const backendpoint = 'http://127.0.0.1:8000'

type DataItem = {
  id: number
  value: number
}

type NewsItem = {
  newsId: string
  category: string
  subcategory: string
  title: string
  abstract: string
  url: string
  title_entities: string
  abstract_entities: string
}

type ApiResponse<T> = {
  data: T[]
  nextCursor: number | null
}

const pageSize = 10

async function fetchData<T>(path: string, pageParam: any) {
  const response = await fetch(
    `${backendpoint}/${path}/?offset=${pageParam}&limit=${pageSize}`,
  )
  if (!response.ok) {
    throw new Error('not ok')
  }
  const data = (await response.json()) as T[]

  const nextCursor = data.length === pageSize ? pageParam + pageSize : null
  return { data, nextCursor }
}

function App() {
 
  return <div className="min-h-screen bg-slate-200 p-20">
    <NewsTable />
  </div>
}

function NewsTable() {
  const {
    data,
    isLoading,
    isError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<ApiResponse<NewsItem>>({
    queryKey: ['data'],
    queryFn: ({ pageParam }) => fetchData('news', pageParam),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    initialPageParam: 0,
  })

  const allItems = data?.pages.flatMap((page) => page.data) || []

  if (!allItems[0]) return <p>No headers</p> 

  const headers1 = Object.keys(allItems[0])
  const headers = headers1 as (keyof NewsItem)[]

  if (isLoading || isError) return <p>Loading or error</p>

  return (
    <div className="flex w-fit flex-col items-center justify-center gap-2">
      <table className="border-collapse text-center">
        <thead>
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                className="border border-slate-400 bg-slate-300 px-8 py-2"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {allItems.map((item, index) => (
            <tr key={item.newsId}>
              {headers.map((h) => (
                <td key={h} className="border border-slate-400 bg-slate-300">
                  {item[h]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {hasNextPage && !isFetchingNextPage && (
        <div className="">
          <button
            onClick={() => fetchNextPage()}
            className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-300"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  )
}

function DataTable() {
  const {
    data,
    isLoading,
    isError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<ApiResponse<DataItem>>({
    queryKey: ['data'],
    queryFn: ({ pageParam }) => fetchData('data', pageParam),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    initialPageParam: 0,
  })

  const allItems = data?.pages.flatMap((page) => page.data) || []

  if (isLoading || isError) return <p>Loading or error</p>

  return (
    <div className="flex w-fit flex-col items-center justify-center gap-2">
      <table className="border-collapse text-center">
        <thead>
          <tr>
            <th className="border border-slate-400 bg-slate-300 px-8 py-2">
              ID
            </th>
            <th className="border border-slate-400 bg-slate-300 px-8 py-2">
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {allItems.map((item) => (
            <tr key={item.id}>
              <td className="border border-slate-400 bg-slate-300">
                {item.id}
              </td>
              <td className="border border-slate-400 bg-slate-300">
                {item.value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {hasNextPage && !isFetchingNextPage && (
        <div className="">
          <button
            onClick={() => fetchNextPage()}
            className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-300"
          >
            Load More
          </button>
        </div>
      )}
    </div>
  )
}

export default App
