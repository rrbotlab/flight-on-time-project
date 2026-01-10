-- tabela perfis
create table if not exists perfis (
    id bigint auto_increment primary key,
    nome varchar(100) not null
);

-- tabela usuários
create table if not exists usuarios (
    id bigint auto_increment primary key,
    nome varchar(100) not null,
    email varchar(100) unique not null,
    senha varchar(255) not null,
    ativo tinyint not null default 1
);

-- tabela usuarios_perfis (usuário pode ter múltiplos perfis)
create table if not exists usuarios_perfis (
    usuario_id bigint,
    perfil_id bigint,
    primary key (usuario_id, perfil_id),
    foreign key (usuario_id) references usuarios(id),
    foreign key (perfil_id) references perfis(id)
);
