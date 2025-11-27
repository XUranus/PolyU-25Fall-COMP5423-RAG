interface SupportModelConfigItem {
    name : string,
    value : string,
    local : boolean // if is a local hugging-face model or a openai API model
}

export const supportModels : SupportModelConfigItem[] = [
    {
        name : "Qwen/Qwen2.5-0.5B-Instruct",
        value : "Qwen/Qwen2.5-0.5B-Instruct",
        local : true,
    },
    {
        name : "Qwen/qwen-turbo",
        value : "qwen-turbo",
        local : false,
    }
]

export default {
    supportModels
}