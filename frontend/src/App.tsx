import { useInfiniteQuery } from '@tanstack/react-query'

const backendpoint = 'http://127.0.0.1:8000'

interface DataItem {
  id: number
  value: number
}

interface DataItem {
  id: number
  value: number
}

interface ApiResponse {
  data: DataItem[]
  nextCursor: number | null
}

const pageSize = 10

const fetchDataPage = async (pageParam: any) => {
  const response = await fetch(
    `${backendpoint}/data/?offset=${pageParam}&limit=${pageSize}`,
  )
  if (!response.ok) {
    throw new Error('Network response was not ok')
  }
  const data = (await response.json()) as DataItem[]

  const nextCursor = data.length === pageSize ? pageParam + pageSize : null
  return { data, nextCursor }
}

function App() {

  const {
    data,
    isLoading,
    isError,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery<ApiResponse>({
    queryKey: ['infiniteData'],
    queryFn: ({ pageParam }) => fetchDataPage(pageParam),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    initialPageParam: 0,
  })

  const allItems = data?.pages.flatMap((page) => page.data) || []

  if (isLoading || isError) return <p>Loading or error</p>

  return (
    <div className="min-h-screen bg-slate-200 p-20">

      <div className='flex flex-col justify-center items-center gap-2 w-fit'>
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
    </div>
  )
}

export default App
